from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tta_utils import average_tta_logits, default_tta_transforms


Tensor = torch.Tensor
TTAFn = Callable[[Tensor], Tensor]
ModelSpecValue = Union[nn.Module, "BaseModelSpec", Tuple[nn.Module, float]]


@dataclass
class BaseModelSpec:
    model: nn.Module
    weight: float = 1.0
    temperature: float = 1.0
    use_tta: bool = True


@dataclass
class BaseModelOutput:
    logits: Tensor
    probabilities: Tensor
    log_probabilities: Tensor
    entropy: Tensor
    normalized_entropy: Tensor
    confidence: Tensor


@dataclass
class BoostingEnsembleOutput:
    output_tensor: Tensor
    predictions: Tensor
    probabilities: Tensor
    log_probabilities: Tensor
    raw_logits: Optional[Tensor]
    aggregation: str
    model_outputs: Dict[str, BaseModelOutput]
    dynamic_weights: Dict[str, Tensor]
    first_pass_probabilities: Tensor
    first_pass_log_probabilities: Tensor


class WeightedBoostingEnsemble(nn.Module):
    """
    Boosting-style weighted combiner for independently trained neural networks.
    """

    def __init__(
        self,
        models: Mapping[str, ModelSpecValue],
        *,
        aggregation: str = "prob",
        uncertainty_penalty: float = 0.25,
        agreement_bonus: float = 0.10,
        clamp_min_weight: float = 1e-6,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if aggregation not in {"prob", "logit"}:
            raise ValueError("aggregation must be either 'prob' or 'logit'.")

        normalized = self._normalize_model_specs(models)

        self.model_names = list(normalized.keys())
        self.models = nn.ModuleDict({name: spec.model for name, spec in normalized.items()})
        self.specs = normalized

        self.aggregation = aggregation
        self.uncertainty_penalty = uncertainty_penalty
        self.agreement_bonus = agreement_bonus
        self.clamp_min_weight = clamp_min_weight
        self.eps = eps

    @staticmethod
    def default_tta_transforms() -> List[TTAFn]:
        return default_tta_transforms()

    @staticmethod
    def _normalize_model_specs(models: Mapping[str, ModelSpecValue]) -> Dict[str, BaseModelSpec]:
        normalized: Dict[str, BaseModelSpec] = {}
        for name, value in models.items():
            if isinstance(value, BaseModelSpec):
                normalized[name] = value
            elif isinstance(value, tuple):
                model, weight = value
                normalized[name] = BaseModelSpec(model=model, weight=float(weight))
            elif isinstance(value, nn.Module):
                normalized[name] = BaseModelSpec(model=value)
            else:
                raise TypeError(
                    f"Unsupported model spec for '{name}'. Use nn.Module, BaseModelSpec, or (model, weight)."
                )

        if not normalized:
            raise ValueError("WeightedBoostingEnsemble requires at least one base model.")

        return normalized

    @staticmethod
    def _entropy(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
        return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)

    @staticmethod
    def _module_device(module: nn.Module) -> Optional[torch.device]:
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter.device

        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer.device

        return None

    @staticmethod
    def _normalize_weights(weight_map: Mapping[str, Tensor], eps: float = 1e-8) -> Dict[str, Tensor]:
        if not weight_map:
            raise ValueError("weight_map must not be empty.")

        total: Optional[Tensor] = None
        for value in weight_map.values():
            total = value if total is None else total + value

        if total is None:
            raise ValueError("weight_map must not be empty.")

        total = total.clamp_min(eps)
        return {name: value / total for name, value in weight_map.items()}

    def _validate_model_device(self, model_name: str, model: nn.Module, images: Tensor) -> None:
        model_device = self._module_device(model)
        if model_device is not None and model_device != images.device:
            raise ValueError(
                f"Base model '{model_name}' is on device {model_device}, but images are on {images.device}."
            )

    def _forward_single_model(
        self,
        model_name: str,
        model: nn.Module,
        images: Tensor,
        *,
        temperature: float,
        tta_transforms: Optional[Sequence[TTAFn]],
        use_tta: bool,
        expected_num_classes: Optional[int],
    ) -> BaseModelOutput:
        self._validate_model_device(model_name, model, images)

        logits = average_tta_logits(
            model,
            images,
            tta_transforms=tta_transforms if use_tta else (lambda batch: batch,),
        )

        if logits.ndim != 2:
            raise ValueError(f"Base model '{model_name}' must return a 2D tensor of shape [batch, classes].")

        if logits.shape[0] != images.shape[0]:
            raise ValueError(f"Base model '{model_name}' returned batch size {logits.shape[0]} for {images.shape[0]} inputs.")

        if expected_num_classes is not None and logits.shape[1] != expected_num_classes:
            raise ValueError(
                f"Base model '{model_name}' returned {logits.shape[1]} classes, expected {expected_num_classes}."
            )

        scaled_logits = logits / max(float(temperature), self.eps)
        probabilities = F.softmax(scaled_logits, dim=1)
        entropy = self._entropy(probabilities, eps=self.eps)
        confidence = probabilities.max(dim=1).values

        num_classes = probabilities.shape[1]
        max_entropy = torch.log(
            torch.tensor(float(max(num_classes, 2)), device=probabilities.device, dtype=probabilities.dtype)
        )
        normalized_entropy = entropy / max_entropy.clamp_min(self.eps)

        return BaseModelOutput(
            logits=scaled_logits,
            probabilities=probabilities,
            log_probabilities=probabilities.clamp_min(self.eps).log(),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            confidence=confidence,
        )

    def _build_output(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> BoostingEnsembleOutput:
        model_outputs: Dict[str, BaseModelOutput] = {}
        dynamic_weights: Dict[str, Tensor] = {}
        expected_num_classes: Optional[int] = None

        for name in self.model_names:
            spec = self.specs[name]
            output = self._forward_single_model(
                name,
                self.models[name],
                images,
                temperature=spec.temperature,
                tta_transforms=tta_transforms,
                use_tta=spec.use_tta,
                expected_num_classes=expected_num_classes,
            )

            if expected_num_classes is None:
                expected_num_classes = output.logits.shape[1]

            model_outputs[name] = output

            base_weight = torch.full_like(output.confidence, fill_value=float(spec.weight), device=images.device)
            uncertainty_factor = 1.0 - self.uncertainty_penalty * output.normalized_entropy
            uncertainty_factor = uncertainty_factor.clamp_min(self.clamp_min_weight)
            dynamic_weights[name] = base_weight * uncertainty_factor

        normalized_first = self._normalize_weights(dynamic_weights, eps=self.eps)

        if self.aggregation == "prob":
            first_pass_probabilities = torch.zeros_like(next(iter(model_outputs.values())).probabilities)
            for name in self.model_names:
                first_pass_probabilities = first_pass_probabilities + (
                    model_outputs[name].probabilities * normalized_first[name].unsqueeze(1)
                )
            first_pass_log_probabilities = first_pass_probabilities.clamp_min(self.eps).log()
        else:
            first_pass_logits = torch.zeros_like(next(iter(model_outputs.values())).logits)
            for name in self.model_names:
                first_pass_logits = first_pass_logits + (model_outputs[name].logits * normalized_first[name].unsqueeze(1))
            first_pass_probabilities = F.softmax(first_pass_logits, dim=1)
            first_pass_log_probabilities = first_pass_probabilities.clamp_min(self.eps).log()

        first_pass_predictions = first_pass_probabilities.argmax(dim=1)
        for name in self.model_names:
            model_predictions = model_outputs[name].probabilities.argmax(dim=1)
            agreement = (model_predictions == first_pass_predictions).float()
            dynamic_weights[name] = dynamic_weights[name] * (1.0 + self.agreement_bonus * agreement)

        normalized_final = self._normalize_weights(dynamic_weights, eps=self.eps)

        if self.aggregation == "prob":
            final_probabilities = torch.zeros_like(first_pass_probabilities)
            for name in self.model_names:
                final_probabilities = final_probabilities + (
                    model_outputs[name].probabilities * normalized_final[name].unsqueeze(1)
                )
            final_log_probabilities = final_probabilities.clamp_min(self.eps).log()
            raw_logits = None
            output_tensor = final_log_probabilities
        else:
            raw_logits = torch.zeros_like(next(iter(model_outputs.values())).logits)
            for name in self.model_names:
                raw_logits = raw_logits + (model_outputs[name].logits * normalized_final[name].unsqueeze(1))
            final_probabilities = F.softmax(raw_logits, dim=1)
            final_log_probabilities = final_probabilities.clamp_min(self.eps).log()
            output_tensor = raw_logits

        return BoostingEnsembleOutput(
            output_tensor=output_tensor,
            predictions=final_probabilities.argmax(dim=1),
            probabilities=final_probabilities,
            log_probabilities=final_log_probabilities,
            raw_logits=raw_logits,
            aggregation=self.aggregation,
            model_outputs=model_outputs,
            dynamic_weights=normalized_final,
            first_pass_probabilities=first_pass_probabilities,
            first_pass_log_probabilities=first_pass_log_probabilities,
        )

    def forward(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> Tensor:
        return self._build_output(images, tta_transforms=tta_transforms).output_tensor

    @torch.no_grad()
    def predict(self, images: Tensor, *, tta_transforms: Optional[Sequence[TTAFn]] = None) -> Tensor:
        return self.predict_with_details(images, tta_transforms=tta_transforms).predictions

    @torch.no_grad()
    def predict_proba(self, images: Tensor, *, tta_transforms: Optional[Sequence[TTAFn]] = None) -> Tensor:
        return self.predict_with_details(images, tta_transforms=tta_transforms).probabilities

    @torch.no_grad()
    def predict_with_details(
        self,
        images: Tensor,
        *,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> BoostingEnsembleOutput:
        return self._build_output(images, tta_transforms=tta_transforms)


@torch.no_grad()
def predict_with_weighted_boosting(
    model_specs: Mapping[str, ModelSpecValue],
    images: Tensor,
    *,
    device: Union[str, torch.device],
    aggregation: str = "prob",
    uncertainty_penalty: float = 0.25,
    agreement_bonus: float = 0.10,
    tta_transforms: Optional[Sequence[TTAFn]] = None,
    return_details: bool = True,
) -> Union[Tensor, BoostingEnsembleOutput]:
    device = torch.device(device)
    ensemble = WeightedBoostingEnsemble(
        model_specs,
        aggregation=aggregation,
        uncertainty_penalty=uncertainty_penalty,
        agreement_bonus=agreement_bonus,
    ).to(device)
    images = images.to(device)
    if return_details:
        return ensemble.predict_with_details(images, tta_transforms=tta_transforms)
    return ensemble.predict(images, tta_transforms=tta_transforms)


__all__ = [
    "BaseModelOutput",
    "BaseModelSpec",
    "BoostingEnsembleOutput",
    "TTAFn",
    "WeightedBoostingEnsemble",
    "predict_with_weighted_boosting",
]
