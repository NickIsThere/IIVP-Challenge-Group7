from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


Tensor = torch.Tensor
TTAFn = Callable[[Tensor], Tensor]


def _identity(images: Tensor) -> Tensor:
    return images


def default_tta_transforms() -> List[TTAFn]:
    return [
        _identity,
        lambda images: TF.affine(images, angle=0, translate=[2, 2], scale=1.0, shear=0),
        lambda images: TF.affine(images, angle=10, translate=[0, 0], scale=1.0, shear=0)
    ]


def resolve_tta_transforms(tta_transforms: Optional[Sequence[TTAFn]]) -> List[TTAFn]:
    transforms = list(tta_transforms) if tta_transforms is not None else default_tta_transforms()
    if not transforms:
        raise ValueError("tta_transforms must not be empty when provided.")
    return transforms


def average_tta_logits(
    model: nn.Module,
    images: Tensor,
    *,
    tta_transforms: Optional[Sequence[TTAFn]] = None,
) -> Tensor:
    logits_sum: Optional[Tensor] = None
    transforms = resolve_tta_transforms(tta_transforms)

    for transform in transforms:
        logits = model(transform(images))
        if logits.ndim != 2:
            raise ValueError("TTA models must return a 2D tensor of shape [batch, classes].")
        logits_sum = logits if logits_sum is None else logits_sum + logits

    if logits_sum is None:
        raise RuntimeError("No logits were produced during TTA aggregation.")

    return logits_sum / len(transforms)


@torch.no_grad()
def soft_voting_probabilities(
    models: Sequence[nn.Module],
    images: Tensor,
    *,
    weights: Optional[Sequence[float]] = None,
    tta_transforms: Optional[Sequence[TTAFn]] = None,
) -> Tensor:
    if not models:
        raise ValueError("soft_voting_probabilities requires at least one model.")

    resolved_weights = list(weights) if weights is not None else [1.0] * len(models)
    if len(resolved_weights) != len(models):
        raise ValueError("weights must have the same length as models.")

    total_weight = float(sum(resolved_weights))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    combined_probabilities: Optional[Tensor] = None
    expected_num_classes: Optional[int] = None

    for model, weight in zip(models, resolved_weights):
        logits = average_tta_logits(model, images, tta_transforms=tta_transforms)
        probabilities = F.softmax(logits, dim=1)

        if expected_num_classes is None:
            expected_num_classes = probabilities.shape[1]
        elif probabilities.shape[1] != expected_num_classes:
            raise ValueError(
                f"Inconsistent class counts in TTA voting: expected {expected_num_classes}, got {probabilities.shape[1]}."
            )

        weighted_probabilities = probabilities * float(weight)
        combined_probabilities = (
            weighted_probabilities
            if combined_probabilities is None
            else combined_probabilities + weighted_probabilities
        )

    if combined_probabilities is None:
        raise RuntimeError("No probabilities were produced during TTA voting.")

    return combined_probabilities / total_weight


__all__ = [
    "TTAFn",
    "average_tta_logits",
    "default_tta_transforms",
    "resolve_tta_transforms",
    "soft_voting_probabilities",
]
