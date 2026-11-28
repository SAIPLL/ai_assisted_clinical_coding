import torch
from omegaconf import OmegaConf


class BaseCallback:
    def __init__(self):
        pass

    def on_initialisation_end(self, trainer=None):
        pass

    def on_train_begin(self, trainer=None):
        pass

    def on_train_end(self, trainer=None):
        pass

    def on_val_begin(self, trainer=None):
        pass

    def on_val_end(self, trainer=None):
        pass

    def on_epoch_begin(self, trainer=None):
        pass

    def on_epoch_end(self, trainer=None):
        pass

    def on_batch_begin(self, trainer=None):
        pass

    def on_batch_end(self, trainer=None):
        pass

    def on_fit_begin(self, trainer=None):
        pass

    def on_fit_end(self, trainer=None):
        pass

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        pass

    def on_end(self, trainer=None):
        pass


class SaveBestModelCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.prev_best = None
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric

    def on_epoch_end(self, trainer=None):
        best_metric = trainer.metric_collections[self.split_name][
            self.target_name
        ].get_best_metric(self.metric_name)
        if self.prev_best is None or best_metric != self.prev_best:
            self.prev_best = best_metric
            trainer.save_checkpoint("best_model.pt")
            print("Saved best model")

    def on_fit_end(self, trainer=None):
        if self.prev_best is not None:
            trainer.load_checkpoint(trainer.experiment_path / "best_model.pt")
            print("Loaded best model")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric
        self.patience = config.patience
        self.counter = 0
        self.prev_best = None

    def on_epoch_end(self, trainer=None):
        """On the end of each epoch, test if the validation metric has improved. If it hasn't improved for self.patience epochs, stop training.

        Args:
            trainer (Trainer, optional): Trainer class. Defaults to None.
        """
        best_metric = trainer.metric_collections[self.split_name][
            self.target_name
        ].get_best_metric(self.metric_name)
        if self.prev_best is None or best_metric > self.prev_best:
            self.prev_best = best_metric
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            trainer.stop_training = True
            print(
                f"Early stopping: {self.counter} epochs without improvement for {self.metric_name}"
            )
