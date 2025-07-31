import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

class Callback:
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_start(self):
        self.all_preds = []
        self.all_labels = []

    def on_epoch_end(self):
        pass

    def on_batch_end(self, train: bool, **kwargs):
        pass

class LogsTracker(Callback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.overall_loss = {"train": [], "eval": []}
        self.overall_metrics = {k: {"train": [], "eval": []} for k in self.trainer.metric_funcs}

    def on_epoch_start(self):
        self.train_epoch_loss = 0
        self.train_num_samples = 0
        self.train_total_metrics = {n:0. for n in self.trainer.metric_funcs}

        self.eval_epoch_loss = 0
        self.eval_num_samples = 0
        self.eval_total_metrics = {n:0. for n in self.trainer.metric_funcs}

    def on_batch_end(self, train: bool, batch_num: int, loss, metrics, batch_size):
        if train:
            self.train_epoch_loss += loss.item()*batch_size
            self.train_num_samples += batch_size

            self.trainer.writer.add_scalar(tag="Loss/train",
                scalar_value=self.train_epoch_loss / self.train_num_samples,
                global_step=self.trainer.epoch * len(self.trainer.train_dataloader) + batch_num)

            if hasattr(self.trainer, "neptune_run"):
                self.trainer.neptune_run["train/Loss"].append(self.train_epoch_loss / self.train_num_samples)

            for k in metrics:
                self.train_total_metrics[k] += batch_size*metrics[k]

            metrics_summary = ""
            for k in self.train_total_metrics:
                v = self.train_total_metrics[k]/self.train_num_samples
                self.trainer.writer.add_scalar(tag=f"{k.title()}/train", scalar_value=v,
                    global_step=self.trainer.epoch * len(self.trainer.train_dataloader) + batch_num)

                if hasattr(self.trainer, "neptune_run"):
                    self.trainer.neptune_run[f"train/{k.title()}"].append(v)
                metrics_summary += f", train_{k}: {v:.3f}"

            self.trainer.logger.info("[Epoch %d/%d] [step %d/%d], train_loss: %5.3f%s" % (
                self.trainer.epoch+1, self.trainer.config["epochs"], batch_num+1,
                len(self.trainer.train_dataloader), self.train_epoch_loss/self.train_num_samples, metrics_summary))
        else:
            self.eval_epoch_loss += loss.item()*batch_size
            self.eval_num_samples += batch_size

            self.trainer.writer.add_scalar(tag="Loss/eval",
                scalar_value=self.eval_epoch_loss / self.eval_num_samples,
                global_step=self.trainer.epoch * len(self.trainer.eval_dataloader) + batch_num)

            if hasattr(self.trainer, "neptune_run"):
                self.trainer.neptune_run["eval/Loss"].append(self.eval_epoch_loss / self.eval_num_samples)

            for k in metrics:
                self.eval_total_metrics[k] += batch_size*metrics[k]

            for k in self.eval_total_metrics:
                v = self.eval_total_metrics[k]/self.eval_num_samples
                self.trainer.writer.add_scalar(tag=f"{k.title()}/eval", scalar_value=v,
                    global_step=self.trainer.epoch * len(self.trainer.eval_dataloader) + batch_num)

                if hasattr(self.trainer, "neptune_run"):
                    self.trainer.neptune_run[f"eval/{k.title()}"].append(v)

    def on_epoch_end(self):
        self.train_epoch_loss /= self.train_num_samples
        self.train_total_metrics = {k: v / self.train_num_samples for k, v in self.train_total_metrics.items()}

        self.eval_epoch_loss /= self.eval_num_samples
        self.eval_total_metrics = {k: v / self.eval_num_samples for k, v in self.eval_total_metrics.items()}

        self.overall_loss["train"].append(self.train_epoch_loss)
        self.overall_loss["eval"].append(self.eval_epoch_loss)

        if hasattr(self.trainer, "neptune_run"):
            self.trainer.neptune_run[f"train/Lr"].append(self.trainer.optimizer.param_groups[0]['lr'])
            # self.trainer.neptune_run["train/LossByEpoch"].append(self.train_epoch_loss)
            # self.trainer.neptune_run["eval/LossByEpoch"].append(self.eval_epoch_loss)
            # for k,v in self.train_total_metrics.items():
            #     self.trainer.neptune_run[f"eval/{k.title()}ByEpoch"].append(v)
            # for k,v in self.eval_total_metrics.items():
            #     self.trainer.neptune_run[f"eval/{k.title()}ByEpoch"].append(v)

        for k in self.overall_metrics:
            self.overall_metrics[k]["train"].append(self.train_total_metrics[k])
            self.overall_metrics[k]["eval"].append(self.eval_total_metrics[k])

        updated = self.trainer.model_checkpoint.update(self.eval_epoch_loss)

        self.trainer.writer.add_scalar(tag="Lr/train", global_step=self.trainer.epoch,
            scalar_value=self.trainer.optimizer.param_groups[0]['lr'])

        self.trainer.logger.info("[End epoch %d/%d], eval_loss: %5.3f, %s, %s %s" % (
            self.trainer.epoch+1, self.trainer.config["epochs"], self.eval_epoch_loss,
            ", ".join([f"eval_{k}: {v:.3f}" for k,v in self.eval_total_metrics.items()]),
            f"lr: {self.trainer.optimizer.param_groups[0]['lr']}",
            "[>> BETTER <<]" if updated else ""))
        