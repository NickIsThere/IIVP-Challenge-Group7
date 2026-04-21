from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from boosting_ensemble import BaseModelSpec, WeightedBoostingEnsemble


Tensor = torch.Tensor


class MetaLearner(nn.Module):
    """
    A small but practical meta-learner for stacking.
    It consumes concatenated base-model outputs and learns how to combine them.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


@dataclass
class StackingBatchOutput:
    predictions: Tensor
    probabilities: Tensor
    logits: Tensor
    meta_features: Tensor


class StackingEnsemble(nn.Module):
    """
    Second-level ensemble that can use the weighted boosting ensemble as a feature source.
    """

    def __init__(
        self,
        base_models: Mapping[str, Union[nn.Module, BaseModelSpec, Tuple[nn.Module, float]]],
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
        self.num_classes = num_classes
        self.include_raw_logits = include_raw_logits
        self.include_probabilities = include_probabilities
        self.include_confidence = include_confidence
        self.include_boosted_output = include_boosted_output

        self.base_ensemble = WeightedBoostingEnsemble(base_models, **(boosting_kwargs or {}))
        self.meta_learner: Optional[MetaLearner] = None
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @torch.no_grad()
    def build_meta_features(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence] = None,
    ) -> Tensor:
        details = self.base_ensemble(images, return_details=True, tta_transforms=tta_transforms)
        model_outputs: Dict[str, Dict[str, Tensor]] = details["model_outputs"]  # type: ignore[assignment]

        features: List[Tensor] = []
        for name in self.base_ensemble.model_names:
            output = model_outputs[name]
            if self.include_raw_logits:
                features.append(output["logits"])
            if self.include_probabilities:
                features.append(output["probabilities"])
            if self.include_confidence:
                features.append(output["confidence"].unsqueeze(1))
                features.append(output["entropy"].unsqueeze(1))

        if self.include_boosted_output:
            features.append(details["logits"])          # type: ignore[arg-type]
            features.append(details["probabilities"])   # type: ignore[arg-type]

        return torch.cat(features, dim=1)

    def initialize_meta_learner(self, feature_dim: int) -> None:
        self.meta_learner = MetaLearner(
            input_dim=feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def fit_meta_learner(
        self,
        train_loader: DataLoader,
        *,
        device: Union[str, torch.device],
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        tta_transforms: Optional[Sequence] = None,
        verbose: bool = True,
    ) -> None:
        device = torch.device(device)
        self.to(device)
        self.base_ensemble.eval()

        # Infer feature size from one batch.
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(device)
        sample_features = self.build_meta_features(sample_images, tta_transforms=tta_transforms)
        self.initialize_meta_learner(sample_features.shape[1])
        assert self.meta_learner is not None
        self.meta_learner.to(device)

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.meta_learner.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                with torch.no_grad():
                    meta_features = self.build_meta_features(images, tta_transforms=tta_transforms)

                logits = self.meta_learner(meta_features)
                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_correct += (logits.argmax(dim=1) == targets).sum().item()
                total += images.size(0)

            if verbose:
                print(
                    f"[Stacking] Epoch {epoch + 1}/{epochs} | "
                    f"loss={running_loss / max(total, 1):.4f} | "
                    f"acc={running_correct / max(total, 1):.4f}"
                )

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence] = None,
    ) -> StackingBatchOutput:
        if self.meta_learner is None:
            raise RuntimeError("Meta learner is not initialized. Call fit_meta_learner(...) first.")

        meta_features = self.build_meta_features(images, tta_transforms=tta_transforms)
        logits = self.meta_learner(meta_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return StackingBatchOutput(
            predictions=predictions,
            probabilities=probabilities,
            logits=logits,
            meta_features=meta_features,
        )


if __name__ == "__main__":

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    stacker = StackingEnsemble(
        # some model
        model = ...,
        num_classes=10,
        include_boosted_output=True,
        boosting_kwargs={"aggregation": "prob", "uncertainty_penalty": 0.2, "agreement_bonus": 0.1},
    ).to(device)

    print(
        "StackingEnsemble created successfully. "
        "Train base models first, then call fit_meta_learner(...) on a held-out loader."
    )
