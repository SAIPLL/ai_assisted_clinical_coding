from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.data import DataLoader

from src import metrics, models
from src.data import datasets, transform
from src.data.datatypes import Data, Lookups
from src.lookups import load_lookups
from src.trainer import callbacks


def get_lookups(
    config: OmegaConf,
    data: Data,
    label_transform: transform.Transform,
    text_transform: transform.Transform,
) -> Lookups:
    return load_lookups(
        config=config,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )


def get_model(config: OmegaConf, data_info: dict) -> models.BaseModel:
    model_class = getattr(models, config.name)
    return model_class(**data_info, **config.configs)


def get_optimizer(config: OmegaConf, model: models.BaseModel) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, config.name)
    return optimizer_class(model.parameters(), **config.configs)


def get_lr_scheduler(
    config: OmegaConf,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if config.name is None:
        return None
    if hasattr(torch.optim.lr_scheduler, config.name):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, config.name)
        return lr_scheduler_class(optimizer, **config.configs)
    from transformers import get_scheduler

    return get_scheduler(
        name=config.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **config.configs,
    )


def get_transform(
    config: OmegaConf,
    targets: set[str] | None = None,
    texts: list[str] | None = None,
    load_transform_path: str | None = None,
) -> transform.Transform:
    """Get transform class

    Args:
        config (OmegaConf): Config for the transform
        targets (Optional[set[str]], optional): Groundtruth targets. Defaults to None.
        texts (Optional[list[str]], optional): Text. If text is not provided, the transform will be fitted on the targets. Defaults to None.
        load_transform_path (Optional[str], optional): Path of where the transform is saved. Will be loaded if not none. Defaults to None.

    Raises:
        ValueError: Error if either texts, targets or load_transform_path are not provided.

    Returns:
        transform.Transform: _description_
    """
    transform_class = getattr(transform, config.name)(**config.configs)
    if load_transform_path:
        transform_class.load(load_transform_path)
        print("loaded transform")

    elif texts:
        transform_class.fit(texts)

    elif targets:
        transform_class.fit(targets)

    else:
        raise ValueError(
            "Provide set of labels, or texts of tokens to perform fit transformation"
        )

    return transform_class


def get_datasets(
    config: OmegaConf,
    data: Data,
    text_transform: transform.Transform,
    label_transform: transform.Transform,
    lookups: Lookups,
) -> dict[str, datasets.BaseDataset]:
    dataset_class = getattr(datasets, config.name)
    datasets_dict = {}

    train_data = data.train
    val_data = data.val
    test_data = data.test
    del data.df

    if config.subset is not None:
        random_indices = torch.randperm(
            min(len(train_data), len(val_data), len(test_data))
        )[: config.subset]
        train_data = [train_data[i] for i in random_indices]
        val_data = [val_data[i] for i in random_indices]
        test_data = [test_data[i] for i in random_indices]

    datasets_dict["train"] = dataset_class(
        train_data,
        split_name="train",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    datasets_dict["val"] = dataset_class(
        val_data,
        split_name="val",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    datasets_dict["test"] = dataset_class(
        test_data,
        split_name="test",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    return datasets_dict


def get_dataloaders(
    config: OmegaConf, datasets_dict: dict[str, datasets.BaseDataset]
) -> dict[str, DataLoader]:
    dataloaders = {}
    train_batch_size = min(config.batch_size, config.max_batch_size)
    pprint(f"Train batch size: {train_batch_size}")
    dataloaders["train"] = DataLoader(
        datasets_dict["train"],
        shuffle=True,
        batch_size=train_batch_size,
        drop_last=config.drop_last,
        collate_fn=datasets_dict["train"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    dataloaders["train_val"] = DataLoader(
        datasets_dict["train"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["val"] = DataLoader(
        datasets_dict["val"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["test"] = DataLoader(
        datasets_dict["test"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["test"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataloaders


def get_metric_collection(
    config: OmegaConf,
    number_of_classes: int,
    code_system_code_indices: torch.Tensor | None = None,
    split_code_indices: torch.Tensor | None = None,
    code_system_name: str | None = None,
) -> metrics.MetricCollection:
    metric_list = []
    for metric in config:
        metric_class = getattr(metrics, metric.name)
        metric_list.append(
            metric_class(number_of_classes=number_of_classes, **metric.configs)
        )

    if code_system_code_indices is not None and split_code_indices is not None:
        # Get overlapping indices
        code_indices = torch.tensor(
            np.intersect1d(code_system_code_indices, split_code_indices)
        )
    elif code_system_code_indices is not None:
        code_indices = code_system_code_indices.clone()
    elif split_code_indices is not None:
        code_indices = split_code_indices.clone()
    else:
        code_indices = None

    return metrics.MetricCollection(
        metrics=metric_list,
        code_indices=code_indices,
        code_system_name=code_system_name,
    )


def get_metric_collections(
    config: OmegaConf,
    number_of_classes: int,
    split_names: tuple[str, ...] = ("train", "train_val", "val", "test"),
    splits_with_multiple_code_systems: frozenset[str] = frozenset(),
    code_system2code_indices: dict[str, torch.Tensor] | None = None,
    split2code_indices: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, metrics.MetricCollection]]:
    metric_collections = defaultdict(dict)
    for split_name in split_names:
        if split2code_indices is not None:
            split_code_indices = split2code_indices.get(split_name)
        else:
            split_code_indices = None

        metric_collections[split_name]["all"] = get_metric_collection(
            config=config,
            number_of_classes=number_of_classes,
            split_code_indices=split_code_indices,
        )

        if split_name not in splits_with_multiple_code_systems:
            continue

        if code_system2code_indices is None:
            continue

        for (
            code_system_name,
            code_system_code_indices,
        ) in code_system2code_indices.items():
            metric_collections[split_name][code_system_name] = get_metric_collection(
                config=config,
                number_of_classes=number_of_classes,
                code_system_code_indices=code_system_code_indices,
                split_code_indices=split_code_indices,
                code_system_name=code_system_name,
            )

    return metric_collections


def get_callbacks(config: OmegaConf) -> list[callbacks.BaseCallback]:
    callbacks_list = []
    for callback in config:
        callback_class = getattr(callbacks, callback.name)
        callbacks_list.append(callback_class(config=callback.configs))
    return callbacks_list
