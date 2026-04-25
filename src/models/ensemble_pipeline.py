from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from src.dataset.augmentation import Augmentation
from src.dataset.data_pipeline import DataPipeline
from src.dataset.trainer import Trainer
from src.models.boosting_ensemble import BaseModelSpec, WeightedBoostingEnsemble
from src.models.factory import Factory
from src.models.stacking_ensemble import MetaLearnerFitResult, StackingEnsemble


Tensor = torch.Tensor


@dataclass
class FoldTrainingResult:
    fold: int
    best_accuracy: float
    history: Dict[str, List[float]]


@dataclass
class BaseModelTrainingResult:
    alias: str
    model_name: str
    model: nn.Module
    fold: int
    best_accuracy: float
    history: Dict[str, List[float]]


@dataclass
class SingleModelExperimentResult:
    kind: str
    model_name: str
    fold_accuracies: List[float]
    mean_accuracy: float


@dataclass
class BoostingExperimentResult:
    kind: str
    base_model_names: List[str]
    fold_accuracies: List[float]
    mean_accuracy: float


@dataclass
class OOFMetaFeatureCollection:
    meta_features: Tensor
    targets: Tensor
    sample_indices: Tensor
    fold_sample_indices: Dict[int, Tuple[int, ...]]
    base_model_accuracies: Dict[str, List[float]]
    reference_base_models: Mapping[str, BaseModelSpec]


@dataclass
class StackingExperimentResult:
    kind: str
    base_model_names: List[str]
    meta_validation_accuracy: float
    meta_validation_fold: int
    meta_feature_shape: Tuple[int, int]
    meta_features: Tensor
    targets: Tensor
    sample_indices: Tuple[int, ...]
    meta_train_sample_indices: Tuple[int, ...]
    meta_validation_sample_indices: Tuple[int, ...]
    fold_sample_indices: Dict[int, Tuple[int, ...]]
    base_model_accuracies: Dict[str, List[float]]
    fit_history: MetaLearnerFitResult
    stacker: StackingEnsemble


@dataclass
class MetaFeatureSplit:
    train_features: Tensor
    train_targets: Tensor
    validation_features: Tensor
    validation_targets: Tensor
    train_sample_indices: Tensor
    validation_sample_indices: Tensor
    validation_fold: int


def default_data_transforms(train_augmentation_prob: float = 0.4) -> Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                Augmentation(prob=train_augmentation_prob),
                transforms.RandomApply([transforms.ElasticTransform(alpha=34.0, sigma=4.0)], p=0.4),
                transforms.ToTensor(),
            ]
        ),
        "validation": transforms.Compose([transforms.ToTensor()]),
    }


def _alias_model_names(model_names: Sequence[str]) -> List[Tuple[str, str]]:
    aliases: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    for model_name in model_names:
        counts[model_name] = counts.get(model_name, 0) + 1
        suffix = counts[model_name]
        alias = model_name if suffix == 1 else f"{model_name}_{suffix}"
        aliases.append((alias, model_name))
    return aliases


def _build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    data_transforms: Dict[str, transforms.Compose],
    batch_size: int,
    num_workers: int,
):
    pipeline = DataPipeline(train_df.reset_index(drop=True), val_df.reset_index(drop=True), transforms=data_transforms)
    return pipeline.get_loaders(batch_size=batch_size, num_workers=num_workers)


def _train_model(
    model_name: str,
    *,
    train_loader,
    val_loader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    model_kwargs: Optional[dict] = None,
) -> Tuple[nn.Module, float, Dict[str, List[float]]]:
    model = Factory.get_model(model_name, num_classes=num_classes, **(model_kwargs or {})).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = Trainer(model, criterion, optimizer, device, scheduler)
    best_accuracy, history = trainer.fit(train_loader, val_loader, epochs)
    return model, best_accuracy, history


def _evaluate_classifier(model: nn.Module, loader, *, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if hasattr(model, "predict"):
                predictions = model.predict(images)
            else:
                predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def _split_oof_meta_features(
    collection: OOFMetaFeatureCollection,
    *,
    meta_validation_fold: int,
) -> MetaFeatureSplit:
    if meta_validation_fold not in collection.fold_sample_indices:
        raise ValueError(f"Fold {meta_validation_fold} is not present in the OOF meta-feature collection.")

    sample_indices = collection.sample_indices
    index_to_position = {
        int(sample_index): position
        for position, sample_index in enumerate(sample_indices.tolist())
    }

    validation_positions = sorted(
        index_to_position[int(sample_index)]
        for sample_index in collection.fold_sample_indices[meta_validation_fold]
    )

    validation_mask = torch.zeros(sample_indices.numel(), dtype=torch.bool)
    validation_mask[validation_positions] = True
    train_mask = ~validation_mask

    if not train_mask.any():
        raise ValueError("Meta-train split is empty. Provide at least one non-validation fold.")
    if not validation_mask.any():
        raise ValueError("Meta-validation split is empty. The held-out fold must contain samples.")

    return MetaFeatureSplit(
        train_features=collection.meta_features[train_mask],
        train_targets=collection.targets[train_mask],
        validation_features=collection.meta_features[validation_mask],
        validation_targets=collection.targets[validation_mask],
        train_sample_indices=sample_indices[train_mask],
        validation_sample_indices=sample_indices[validation_mask],
        validation_fold=meta_validation_fold,
    )


def run_single_model_cv(
    df: pd.DataFrame,
    model_name: str,
    *,
    folds: Sequence[int],
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    data_transforms: Dict[str, transforms.Compose],
    device: torch.device,
    model_kwargs: Optional[dict] = None,
) -> SingleModelExperimentResult:
    fold_accuracies: List[float] = []

    for fold in folds:
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        train_loader, val_loader = _build_loaders(
            train_df,
            val_df,
            data_transforms=data_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        _, best_accuracy, _ = _train_model(
            model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            epochs=epochs,
            model_kwargs=model_kwargs,
        )
        fold_accuracies.append(best_accuracy)

    mean_accuracy = sum(fold_accuracies) / max(len(fold_accuracies), 1)
    return SingleModelExperimentResult(
        kind="single",
        model_name=model_name,
        fold_accuracies=fold_accuracies,
        mean_accuracy=mean_accuracy,
    )


def _train_base_models_for_fold(
    df: pd.DataFrame,
    *,
    fold: int,
    aliased_model_names: Sequence[Tuple[str, str]],
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    data_transforms: Dict[str, transforms.Compose],
    device: torch.device,
    model_kwargs_by_name: Optional[Mapping[str, dict]] = None,
) -> Tuple[List[BaseModelTrainingResult], object]:
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]
    train_loader, val_loader = _build_loaders(
        train_df,
        val_df,
        data_transforms=data_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    training_results: List[BaseModelTrainingResult] = []
    for alias, model_name in aliased_model_names:
        model, best_accuracy, history = _train_model(
            model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            epochs=epochs,
            model_kwargs=(model_kwargs_by_name or {}).get(model_name),
        )
        training_results.append(
            BaseModelTrainingResult(
                alias=alias,
                model_name=model_name,
                model=model,
                fold=fold,
                best_accuracy=best_accuracy,
                history=history,
            )
        )

    return training_results, val_loader


def run_boosting_cv(
    df: pd.DataFrame,
    base_model_names: Sequence[str],
    *,
    folds: Sequence[int],
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    data_transforms: Dict[str, transforms.Compose],
    device: torch.device,
    boosting_kwargs: Optional[dict] = None,
    model_kwargs_by_name: Optional[Mapping[str, dict]] = None,
) -> BoostingExperimentResult:
    aliased_model_names = _alias_model_names(base_model_names)
    fold_accuracies: List[float] = []

    for fold in folds:
        training_results, val_loader = _train_base_models_for_fold(
            df,
            fold=fold,
            aliased_model_names=aliased_model_names,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            data_transforms=data_transforms,
            device=device,
            model_kwargs_by_name=model_kwargs_by_name,
        )

        model_specs = {
            result.alias: BaseModelSpec(model=result.model)
            for result in training_results
        }
        ensemble = WeightedBoostingEnsemble(model_specs, **(boosting_kwargs or {})).to(device)
        fold_accuracies.append(_evaluate_classifier(ensemble, val_loader, device=device))

    mean_accuracy = sum(fold_accuracies) / max(len(fold_accuracies), 1)
    return BoostingExperimentResult(
        kind="boosting",
        base_model_names=list(base_model_names),
        fold_accuracies=fold_accuracies,
        mean_accuracy=mean_accuracy,
    )


def collect_oof_meta_features(
    df: pd.DataFrame,
    base_model_names: Sequence[str],
    *,
    folds: Sequence[int],
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    data_transforms: Dict[str, transforms.Compose],
    device: torch.device,
    boosting_kwargs: Optional[dict] = None,
    stacking_kwargs: Optional[dict] = None,
    model_kwargs_by_name: Optional[Mapping[str, dict]] = None,
) -> OOFMetaFeatureCollection:
    aliased_model_names = _alias_model_names(base_model_names)
    feature_chunks: List[Tensor] = []
    target_chunks: List[Tensor] = []
    sample_index_chunks: List[Tensor] = []
    fold_sample_indices: Dict[int, Tuple[int, ...]] = {}
    base_model_accuracies = {alias: [] for alias, _ in aliased_model_names}
    reference_base_models: Optional[Mapping[str, BaseModelSpec]] = None

    for fold in folds:
        fold_val_df = df[df["fold"] == fold]
        original_indices = tuple(int(index) for index in fold_val_df.index.tolist())
        training_results, val_loader = _train_base_models_for_fold(
            df,
            fold=fold,
            aliased_model_names=aliased_model_names,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            data_transforms=data_transforms,
            device=device,
            model_kwargs_by_name=model_kwargs_by_name,
        )

        model_specs = {result.alias: BaseModelSpec(model=result.model) for result in training_results}
        reference_base_models = model_specs
        for result in training_results:
            base_model_accuracies[result.alias].append(result.best_accuracy)

        stacker = StackingEnsemble(
            model_specs,
            num_classes=num_classes,
            boosting_kwargs=boosting_kwargs,
            **(stacking_kwargs or {}),
        ).to(device)
        stacker.eval()

        cursor = 0
        fold_meta_features: List[Tensor] = []
        fold_targets: List[Tensor] = []
        fold_indices: List[int] = []

        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device)
                batch_size_current = targets.size(0)
                meta_features = stacker.build_meta_features(images)

                fold_meta_features.append(meta_features.cpu())
                fold_targets.append(targets.cpu())
                fold_indices.extend(original_indices[cursor:cursor + batch_size_current])
                cursor += batch_size_current

        feature_chunks.append(torch.cat(fold_meta_features, dim=0))
        target_chunks.append(torch.cat(fold_targets, dim=0))
        sample_index_chunks.append(torch.tensor(fold_indices, dtype=torch.long))
        fold_sample_indices[fold] = tuple(fold_indices)

    if reference_base_models is None:
        raise ValueError("No folds were processed while collecting OOF meta features.")

    meta_features = torch.cat(feature_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    sample_indices = torch.cat(sample_index_chunks, dim=0)

    sorted_positions = torch.argsort(sample_indices)
    meta_features = meta_features[sorted_positions]
    targets = targets[sorted_positions]
    sample_indices = sample_indices[sorted_positions]

    if sample_indices.numel() != len(df):
        raise ValueError("OOF meta-feature collection did not produce exactly one row per source sample.")
    if sample_indices.unique().numel() != sample_indices.numel():
        raise ValueError("OOF meta-feature collection produced duplicate sample indices.")

    return OOFMetaFeatureCollection(
        meta_features=meta_features,
        targets=targets,
        sample_indices=sample_indices,
        fold_sample_indices=fold_sample_indices,
        base_model_accuracies=base_model_accuracies,
        reference_base_models=reference_base_models,
    )


def run_stacking_oof_cv(
    df: pd.DataFrame,
    base_model_names: Sequence[str],
    *,
    folds: Sequence[int],
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    data_transforms: Dict[str, transforms.Compose],
    device: torch.device,
    boosting_kwargs: Optional[dict] = None,
    stacking_kwargs: Optional[dict] = None,
    meta_fit_kwargs: Optional[dict] = None,
    model_kwargs_by_name: Optional[Mapping[str, dict]] = None,
) -> StackingExperimentResult:
    if len(folds) < 2:
        raise ValueError("Stacking requires at least 2 folds so one fold can be held out for meta-validation.")

    oof_collection = collect_oof_meta_features(
        df,
        base_model_names,
        folds=folds,
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        data_transforms=data_transforms,
        device=device,
        boosting_kwargs=boosting_kwargs,
        stacking_kwargs=stacking_kwargs,
        model_kwargs_by_name=model_kwargs_by_name,
    )
    meta_validation_fold = int(folds[-1])
    meta_split = _split_oof_meta_features(
        oof_collection,
        meta_validation_fold=meta_validation_fold,
    )

    stacker = StackingEnsemble(
        oof_collection.reference_base_models,
        num_classes=num_classes,
        boosting_kwargs=boosting_kwargs,
        **(stacking_kwargs or {}),
    )
    resolved_meta_fit_kwargs = {"batch_size": batch_size, "verbose": False}
    resolved_meta_fit_kwargs.update(meta_fit_kwargs or {})
    fit_history = stacker.fit_meta_learner(
        meta_split.train_features,
        meta_split.train_targets,
        device=device,
        **resolved_meta_fit_kwargs,
    )

    predictions = stacker.predict_from_meta_features(meta_split.validation_features.to(device)).predictions.cpu()
    meta_validation_accuracy = (predictions == meta_split.validation_targets).float().mean().item()

    return StackingExperimentResult(
        kind="stacking",
        base_model_names=list(base_model_names),
        meta_validation_accuracy=meta_validation_accuracy,
        meta_validation_fold=meta_split.validation_fold,
        meta_feature_shape=tuple(oof_collection.meta_features.shape),
        meta_features=oof_collection.meta_features,
        targets=oof_collection.targets,
        sample_indices=tuple(int(index) for index in oof_collection.sample_indices.tolist()),
        meta_train_sample_indices=tuple(int(index) for index in meta_split.train_sample_indices.tolist()),
        meta_validation_sample_indices=tuple(int(index) for index in meta_split.validation_sample_indices.tolist()),
        fold_sample_indices=oof_collection.fold_sample_indices,
        base_model_accuracies=oof_collection.base_model_accuracies,
        fit_history=fit_history,
        stacker=stacker,
    )


__all__ = [
    "BoostingExperimentResult",
    "OOFMetaFeatureCollection",
    "SingleModelExperimentResult",
    "StackingExperimentResult",
    "collect_oof_meta_features",
    "default_data_transforms",
    "run_boosting_cv",
    "run_single_model_cv",
    "run_stacking_oof_cv",
]
