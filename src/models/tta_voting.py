from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .tta_utils import TTAFn, soft_voting_probabilities


def tta_voting(
    models_list: Sequence[nn.Module],
    weights_or_images: Union[Sequence[float], torch.Tensor],
    images_or_device: Optional[Union[torch.Tensor, torch.device]] = None,
    device: Optional[torch.device] = None,
    tta_transforms: Optional[Sequence[TTAFn]] = None,
) -> torch.Tensor:
    if torch.is_tensor(weights_or_images):
        images = weights_or_images
        weights_list = None
        if images_or_device is not None:
            device = torch.device(images_or_device)
    else:
        weights_list = weights_or_images
        if images_or_device is None or not torch.is_tensor(images_or_device):
            raise ValueError("images must be provided when weights are passed explicitly.")
        images = images_or_device

    if device is not None:
        images = images.to(device)

    probabilities = soft_voting_probabilities(
        models_list,
        images,
        weights=weights_list,
        tta_transforms=tta_transforms,
    )
    return probabilities.argmax(dim=1)
