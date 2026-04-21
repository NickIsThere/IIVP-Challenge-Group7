from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .boosting_ensemble import BaseModelSpec, BoostingEnsembleOutput, TTAFn, WeightedBoostingEnsemble


Tensor = torch.Tensor
ModelSpecValue = Union[nn.Module, BaseModelSpec, Tuple[nn.Module, float]]


class MetaLearner(nn.Module):
    """
    Small meta-learner for second-level stacking.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()

        reduced_dim = max(hidden_dim // 2, 1)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, reduced_dim),
            nn.ReLU(),
            nn.LayerNorm(reduced_dim),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


@dataclass
class MetaLearnerFitResult:
    train_loss: List[float]
    train_accuracy: List[float]


@dataclass
class StackingBatchOutput:
    predictions: Tensor
    probabilities: Tensor
    logits: Tensor
    meta_features: Tensor


class StackingEnsemble(nn.Module):
    """
    Second-level ensemble that consumes frozen base-model outputs.
    """

    def __init__(
        self,
        base_models: Mapping[str, ModelSpecValue],
        *,
        num_classes: int,
        include_raw_logits: bool = True,
        include_probabilities: bool = True,
        include_confidence: bool = True,
        include_boosted_output: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        boosting_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if not any((include_raw_logits, include_probabilities, include_confidence, include_boosted_output)):
            raise ValueError(
                "Enable at least one meta-feature source: raw logits, probabilities, confidence, or boosted output."
            )

        self.num_classes = num_classes
        self.include_raw_logits = include_raw_logits
        self.include_probabilities = include_probabilities
        self.include_confidence = include_confidence
        self.include_boosted_output = include_boosted_output
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.base_ensemble = WeightedBoostingEnsemble(base_models, **(boosting_kwargs or {}))
        self.meta_learner: Optional[MetaLearner] = None

    @staticmethod
    def _meta_learner_device(meta_learner: nn.Module) -> Optional[torch.device]:
        parameter = next(meta_learner.parameters(), None)
        if parameter is not None:
            return parameter.device

        buffer = next(meta_learner.buffers(), None)
        if buffer is not None:
            return buffer.device

        return None

    def _require_meta_learner(self) -> MetaLearner:
        if self.meta_learner is None:
            raise RuntimeError("Meta learner is not initialized. Call fit_meta_learner(...) first.")
        return self.meta_learner

    def _validate_meta_learner_device(self, meta_features: Tensor) -> None:
        meta_learner = self._require_meta_learner()
        model_device = self._meta_learner_device(meta_learner)
        if model_device is not None and model_device != meta_features.device:
            raise ValueError(
                f"Meta learner is on device {model_device}, but meta features are on {meta_features.device}."
            )

    def _extract_features(self, details: BoostingEnsembleOutput) -> Tensor:
        features: List[Tensor] = []

        for name in self.base_ensemble.model_names:
            output = details.model_outputs[name]

            if self.include_raw_logits:
                features.append(output.logits)

            if self.include_probabilities:
                features.append(output.probabilities)

            if self.include_confidence:
                features.append(output.confidence.unsqueeze(1))
                features.append(output.entropy.unsqueeze(1))
                features.append(output.normalized_entropy.unsqueeze(1))

        if self.include_boosted_output:
            features.append(details.output_tensor)
            features.append(details.probabilities)

        return torch.cat(features, dim=1)

    @torch.no_grad()
    def build_meta_features(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> Tensor:
        details = self.base_ensemble.predict_with_details(images, tta_transforms=tta_transforms)
        return self._extract_features(details)

    def initialize_meta_learner(self, feature_dim: int) -> None:
        self.meta_learner = MetaLearner(
            input_dim=feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    @staticmethod
    def _meta_feature_loader(
        features_or_loader: Union[Tensor, DataLoader],
        targets: Optional[Tensor],
        batch_size: int,
    ) -> Tuple[DataLoader, Tensor]:
        if isinstance(features_or_loader, DataLoader):
            loader = features_or_loader
            try:
                sample_features, sample_targets = next(iter(loader))
            except StopIteration as exc:
                raise ValueError("Meta-feature loader is empty. Cannot fit meta learner.") from exc

            if sample_features.ndim != 2:
                raise ValueError("Meta features must be a 2D tensor of shape [batch, feature_dim].")
            if sample_targets.ndim != 1:
                raise ValueError("Targets must be a 1D tensor of class indices.")
            return loader, sample_features

        if targets is None:
            raise ValueError("targets must be provided when fitting the meta learner from feature tensors.")

        if features_or_loader.ndim != 2:
            raise ValueError("Meta features must be a 2D tensor of shape [batch, feature_dim].")
        if targets.ndim != 1:
            raise ValueError("Targets must be a 1D tensor of class indices.")
        if features_or_loader.shape[0] != targets.shape[0]:
            raise ValueError("Meta features and targets must contain the same number of samples.")

        dataset = TensorDataset(features_or_loader, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), features_or_loader

    def _logits_from_meta_features(self, meta_features: Tensor) -> Tensor:
        self._validate_meta_learner_device(meta_features)
        meta_learner = self._require_meta_learner()
        return meta_learner(meta_features)

    def fit_meta_learner(
        self,
        features_or_loader: Union[Tensor, DataLoader],
        targets: Optional[Tensor] = None,
        *,
        device: Union[str, torch.device],
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> MetaLearnerFitResult:
        device = torch.device(device)
        loader, sample_features = self._meta_feature_loader(features_or_loader, targets, batch_size=batch_size)

        self.initialize_meta_learner(sample_features.shape[1])
        meta_learner = self._require_meta_learner()
        meta_learner.to(device)

        optimizer = torch.optim.Adam(meta_learner.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = MetaLearnerFitResult(train_loss=[], train_accuracy=[])

        for epoch in range(epochs):
            meta_learner.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                logits = meta_learner(batch_features)
                loss = criterion(logits, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_features.size(0)
                running_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
                total += batch_features.size(0)

            epoch_loss = running_loss / max(total, 1)
            epoch_accuracy = running_correct / max(total, 1)
            history.train_loss.append(epoch_loss)
            history.train_accuracy.append(epoch_accuracy)

            if verbose:
                print(
                    f"[Stacking] Epoch {epoch + 1}/{epochs} | "
                    f"loss={epoch_loss:.4f} | acc={epoch_accuracy:.4f}"
                )

        return history

    def forward(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> Tensor:
        meta_features = self.build_meta_features(images, tta_transforms=tta_transforms)
        return self._logits_from_meta_features(meta_features)

    @torch.no_grad()
    def predict_from_meta_features(self, meta_features: Tensor) -> StackingBatchOutput:
        logits = self._logits_from_meta_features(meta_features)
        probabilities = F.softmax(logits, dim=1)
        return StackingBatchOutput(
            predictions=probabilities.argmax(dim=1),
            probabilities=probabilities,
            logits=logits,
            meta_features=meta_features,
        )

    @torch.no_grad()
    def predict_with_details(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> StackingBatchOutput:
        meta_features = self.build_meta_features(images, tta_transforms=tta_transforms)
        return self.predict_from_meta_features(meta_features)


__all__ = [
    "MetaLearner",
    "MetaLearnerFitResult",
    "StackingBatchOutput",
    "StackingEnsemble",
]
