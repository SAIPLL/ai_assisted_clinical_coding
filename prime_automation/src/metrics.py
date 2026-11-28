from copy import deepcopy
from typing import ClassVar

import torch

from src.utils import detach


class Metric:
    base_tags: ClassVar[set[str]] = set()
    _str_value_fmt = "<.3"
    higher_is_better = True
    batch_update = True
    filter_codes = True

    def __init__(
        self,
        name: str,
        tags: set[str] | None,
        number_of_classes: int,
        threshold: float | None = None,
    ):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.number_of_classes = number_of_classes
        self.device = "cpu"
        self.threshold = threshold
        self.reset()

    def update(self, batch: dict):
        """Update the metric from a batch"""
        raise NotImplementedError()

    def set_target_boolean_indices(self, target_boolean_indices: list[bool]):
        self.target_boolean_indices = target_boolean_indices

    def compute(self):
        """Compute the metric value"""
        raise NotImplementedError()

    def reset(self):
        """Reset the metric"""
        raise NotImplementedError()

    def to(self, device: str):
        self.device = device
        if self.threshold is not None:
            self.threshold = torch.tensor(self.threshold).clone().to(device)
        self.reset()
        return self

    def copy(self):
        return deepcopy(self)

    def set_number_of_classes(self, number_of_classes: int):
        self.number_of_classes = number_of_classes


class MetricCollection:
    def __init__(
        self,
        metrics: list[Metric],
        code_indices: torch.Tensor | None = None,
        code_system_name: str | None = None,
    ):
        self.metrics = metrics
        self.code_system_name = code_system_name
        if code_indices is not None:
            # Get overlapping indices
            self.code_indices = code_indices.clone()
            self.set_number_of_classes(len(code_indices))
        else:
            self.code_indices = None
        self.reset()

    def set_number_of_classes(self, number_of_classes_split: int):
        """Sets the number of classes for metrics with the filter_codes attribute to the number of classes in the split.
        Args:
            number_of_classes_split (int): Number of classes in the split
        """
        for metric in self.metrics:
            if metric.filter_codes:
                metric.set_number_of_classes(number_of_classes_split)

    def to(self, device: str):
        self.metrics = [metric.to(device) for metric in self.metrics]
        if self.code_indices is not None:
            self.code_indices = self.code_indices.to(device)
        return self

    def filter_batch(self, batch: dict) -> dict:
        if self.code_indices is None:
            return batch

        filter_batch = {}
        targets, logits = batch["targets"], batch["logits"]

        filtered_targets = torch.index_select(targets, -1, self.code_indices)
        filtered_logits = torch.index_select(logits, -1, self.code_indices)
        # Elements in the batch with targets
        idx_targets = torch.sum(filtered_targets, dim=-1) > 0
        # Remove all elements wihtout targets
        filter_batch["targets"] = filtered_targets[idx_targets]
        filter_batch["logits"] = filtered_logits[idx_targets]
        return filter_batch

    def filter_tensor(
        self, tensor: torch.Tensor, code_indices: torch.Tensor
    ) -> list[torch.Tensor]:
        if code_indices is None:
            return tensor
        return torch.index_select(tensor, -1, code_indices)

    def is_best(
        self,
        prev_best: torch.Tensor | None,
        current: torch.Tensor,
        higher_is_better: bool,
    ) -> bool:
        if higher_is_better:
            return prev_best is None or current > prev_best
        else:
            return prev_best is None or current < prev_best

    def update_best_metrics(self, metric_dict: dict[str, torch.Tensor]):
        for metric in self.metrics:
            if metric.name not in metric_dict:
                continue

            if self.is_best(
                self.best_metrics[metric.name],
                metric_dict[metric.name],
                metric.higher_is_better,
            ):
                self.best_metrics[metric.name] = metric_dict[metric.name]

    def update(self, batch: dict):
        for metric in self.metrics:
            if metric.batch_update and not metric.filter_codes:
                metric.update(batch)

        filtered_batch = self.filter_batch(batch)

        for metric in self.metrics:
            if metric.batch_update and metric.filter_codes:
                metric.update(filtered_batch)

    def compute(
        self,
        logits: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        metric_dict = {
            metric.name: metric.compute()
            for metric in self.metrics
            if metric.batch_update
        }
        if logits is not None and targets is not None:
            # Compute the metrics for the whole dataset
            if self.code_indices is not None:
                logits_filtered = self.filter_tensor(logits, self.code_indices.cpu())
                targets_filtered = self.filter_tensor(targets, self.code_indices.cpu())

            for metric in self.metrics:
                if metric.batch_update:
                    continue
                if metric.filter_codes and self.code_indices is not None:
                    metric_dict[metric.name] = metric.compute(
                        logits=logits_filtered, targets=targets_filtered
                    )
                else:
                    metric_dict[metric.name] = metric.compute(
                        logits=logits, targets=targets
                    )

            metric_dict.update(
                {
                    metric.name: metric.compute(logits=logits, targets=targets)
                    for metric in self.metrics
                    if not metric.batch_update
                }
            )
        self.update_best_metrics(metric_dict)
        return metric_dict

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def reset(self):
        self.reset_metrics()
        self.best_metrics = {metric.name: None for metric in self.metrics}

    def get_best_metric(self, metric_name: str) -> dict[str, torch.Tensor]:
        return self.best_metrics[metric_name]

    def copy(self):
        return deepcopy(self)

    def set_threshold(self, threshold: float):
        for metric in self.metrics:
            if hasattr(metric, "threshold"):
                metric.threshold = threshold


""" ------------Information Retrieval Metrics-------------"""


class Precision_K(Metric):
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        k: int = 10,
        name: str = "precision",
        tags: set[str] | None = None,
        number_of_classes: int | None = None,
        filter_codes: bool = True,
    ):
        name = f"{name}@{k}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self._k = k
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        top_k = torch.topk(logits, dim=1, k=self._k)

        targets_k = targets.gather(1, top_k.indices)
        logits_k = torch.ones(targets_k.shape, device=targets_k.device)

        tp = torch.sum(logits_k * targets_k, dim=1)
        fp = torch.sum((logits_k) * (1 - targets_k), dim=1)
        self._num_examples += logits.size(0)
        self._precision_sum += torch.sum(tp / (tp + fp + 1e-10))

    def compute(self) -> torch.Tensor:
        return self._precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._precision_sum = torch.tensor(0.0).to(self.device)
