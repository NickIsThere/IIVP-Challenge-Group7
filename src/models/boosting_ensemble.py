from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
ModelDict = Mapping[str, nn.Module]
WeightDict = Mapping[str, float]
TTAFn = Callable[[Tensor], Tensor]


@dataclass
class BaseModelSpec:
    model: nn.Module
    weight: float = 1.0
    temperature: float = 1.0
    use_tta: bool = True


class WeightedBoostingEnsemble(nn.Module):
    """
    Features:
    - accepts a dict of model specs: {name: BaseModelSpec(...)}
    - weighted aggregation in probability or logit space
    - per-model temperature scaling
    - optional uncertainty penalty (downweights uncertain models per batch)
    - optional agreement bonus (upweights models that agree with the ensemble)
    - optional TTA support
    - returns both predictions and rich diagnostics

    This is not gradient boosting in the classical sequential sense. Instead, it is a
    boosting-style weighted combiner for independently trained neural networks.
    """

    def __init__(
        self,
        models: Mapping[str, Union[nn.Module, BaseModelSpec, Tuple[nn.Module, float]]],
        *,
        aggregation: str = "prob",
        uncertainty_penalty: float = 0.25,
        agreement_bonus: float = 0.10,
        clamp_min_weight: float = 1e-6,
    ) -> None:
        super().__init__()

        if aggregation not in {"prob", "logit"}:
            raise ValueError("aggregation must be either 'prob' or 'logit'.")

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

        self.model_names = list(normalized.keys())
        self.models = nn.ModuleDict({name: spec.model for name, spec in normalized.items()})
        self.specs = normalized
        self.aggregation = aggregation
        self.uncertainty_penalty = uncertainty_penalty
        self.agreement_bonus = agreement_bonus
        self.clamp_min_weight = clamp_min_weight

    @staticmethod
    def default_tta_transforms() -> List[TTAFn]:
        return [
            lambda x: x,
            lambda x: torch.flip(x, dims=[-1]),
            lambda x: torch.flip(x, dims=[-2]),
        ]

    @staticmethod
    def _entropy(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
        return -(probabilities * (probabilities.clamp_min(eps).log())).sum(dim=1)

    @staticmethod
    def _normalize_weights(weight_map: MutableMapping[str, Tensor], eps: float = 1e-8) -> MutableMapping[str, Tensor]:
        total = None
        for value in weight_map.values():
            total = value if total is None else total + value
        assert total is not None
        total = total.clamp_min(eps)
        return {k: v / total for k, v in weight_map.items()}

    def _forward_single_model(
        self,
        model: nn.Module,
        images: Tensor,
        *,
        temperature: float,
        tta_transforms: Optional[Sequence[TTAFn]],
        use_tta: bool,
    ) -> Dict[str, Tensor]:
        logits_accum = None
        transforms = list(tta_transforms) if (tta_transforms is not None and use_tta) else [lambda x: x]

        for transform in transforms:
            augmented = transform(images)
            logits = model(augmented) / max(temperature, 1e-8)
            logits_accum = logits if logits_accum is None else logits_accum + logits

        assert logits_accum is not None
        mean_logits = logits_accum / len(transforms)
        mean_prob = F.softmax(mean_logits, dim=1)
        entropy = self._entropy(mean_prob)
        confidence = mean_prob.max(dim=1).values

        return {
            "logits": mean_logits,
            "probabilities": mean_prob,
            "entropy": entropy,
            "confidence": confidence,
        }

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        return_details: bool = True,
        tta_transforms: Optional[Sequence[TTAFn]] = None,
    ) -> Union[Tensor, Dict[str, Tensor], Dict[str, object]]:
        device = images.device
        model_outputs: Dict[str, Dict[str, Tensor]] = {}
        dynamic_weights: Dict[str, Tensor] = {}

        for name in self.model_names:
            spec = self.specs[name]
            model = self.models[name]
            model.eval()

            output = self._forward_single_model(
                model,
                images,
                temperature=spec.temperature,
                tta_transforms=tta_transforms,
                use_tta=spec.use_tta,
            )
            model_outputs[name] = output

            base = torch.full_like(output["confidence"], fill_value=float(spec.weight), device=device)
            uncertainty_factor = 1.0 - self.uncertainty_penalty * output["entropy"]
            uncertainty_factor = uncertainty_factor.clamp_min(self.clamp_min_weight)
            dynamic_weights[name] = base * uncertainty_factor

        # First pass ensemble for agreement scoring.
        normalized_first = self._normalize_weights(dynamic_weights)

        if self.aggregation == "prob":
            ensemble_prob = None
            for name in self.model_names:
                weighted = model_outputs[name]["probabilities"] * normalized_first[name].unsqueeze(1)
                ensemble_prob = weighted if ensemble_prob is None else ensemble_prob + weighted
            assert ensemble_prob is not None
            ensemble_logits = ensemble_prob.clamp_min(1e-8).log()
        else:
            ensemble_logits = None
            for name in self.model_names:
                weighted = model_outputs[name]["logits"] * normalized_first[name].unsqueeze(1)
                ensemble_logits = weighted if ensemble_logits is None else ensemble_logits + weighted
            assert ensemble_logits is not None
            ensemble_prob = F.softmax(ensemble_logits, dim=1)

        # Agreement bonus: boost models whose predictions align with first-pass ensemble.
        ensemble_pred = ensemble_prob.argmax(dim=1)
        for name in self.model_names:
            model_pred = model_outputs[name]["probabilities"].argmax(dim=1)
            agreement = (model_pred == ensemble_pred).float()
            dynamic_weights[name] = dynamic_weights[name] * (1.0 + self.agreement_bonus * agreement)

        normalized_final = self._normalize_weights(dynamic_weights)

        if self.aggregation == "prob":
            final_prob = None
            for name in self.model_names:
                weighted = model_outputs[name]["probabilities"] * normalized_final[name].unsqueeze(1)
                final_prob = weighted if final_prob is None else final_prob + weighted
            assert final_prob is not None
            final_logits = final_prob.clamp_min(1e-8).log()
        else:
            final_logits = None
            for name in self.model_names:
                weighted = model_outputs[name]["logits"] * normalized_final[name].unsqueeze(1)
                final_logits = weighted if final_logits is None else final_logits + weighted
            assert final_logits is not None
            final_prob = F.softmax(final_logits, dim=1)

        predictions = final_prob.argmax(dim=1)

        if not return_details:
            return predictions

        details: Dict[str, object] = {
            "predictions": predictions,
            "probabilities": final_prob,
            "logits": final_logits,
            "model_outputs": model_outputs,
            "dynamic_weights": normalized_final,
        }
        return details


@torch.no_grad()
def predict_with_weighted_boosting(
    model_specs: Mapping[str, Union[nn.Module, BaseModelSpec, Tuple[nn.Module, float]]],
    images: Tensor,
    *,
    device: Union[str, torch.device],
    aggregation: str = "prob",
    uncertainty_penalty: float = 0.25,
    agreement_bonus: float = 0.10,
    tta_transforms: Optional[Sequence[TTAFn]] = None,
    return_details: bool = True,
) -> Union[Tensor, Dict[str, object]]:
    """
    Convenience function if you prefer a functional API.
    """
    images = images.to(device)
    ensemble = WeightedBoostingEnsemble(
        model_specs,
        aggregation=aggregation,
        uncertainty_penalty=uncertainty_penalty,
        agreement_bonus=agreement_bonus,
    ).to(device)
    return ensemble(images, return_details=return_details, tta_transforms=tta_transforms)


if __name__ == "__main__":
    # Example usage.

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models = {
        "cnn_a": BaseModelSpec(SOMEMODEL().to(device), weight=1.2, temperature=1.0),
        "cnn_b": BaseModelSpec(SOMEMODEL().to(device), weight=0.9, temperature=1.1),
        "cnn_c": BaseModelSpec(SOMEMODEL().to(device), weight=1.4, temperature=0.95),
    }

    result = predict_with_weighted_boosting(
        models,
        x,
        device=device,
        tta_transforms=WeightedBoostingEnsemble.default_tta_transforms(),
    )
    print("Predictions:", result["predictions"])
    print("Dynamic weights:", {k: v[:3].tolist() for k, v in result["dynamic_weights"].items()})
