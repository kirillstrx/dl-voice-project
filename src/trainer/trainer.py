import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]

        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad(set_to_none=True)

        batch.update(self.model(**batch))
        batch.update(self.criterion(**batch))

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for metric in metric_funcs:
            if not getattr(metric, "requires_full_dataset", False):
                metrics.update(metric.name, metric(**batch))

        return batch

    def _train_epoch(self, epoch):
        logs = super()._train_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        all_logits = []
        all_labels = []

        with torch.inference_mode():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, self.evaluation_metrics)

                all_logits.append(batch["logits"].detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        for metric in self.metrics["inference"]:
            if getattr(metric, "requires_full_dataset", False):
                self.evaluation_metrics.update(metric.name, metric(logits=logits, labels=labels))

        self.writer.set_step(epoch * self.epoch_len, part)
        self._log_scalars(self.evaluation_metrics)
        self._log_batch(batch_idx, batch, part)

        return self.evaluation_metrics.result()

    def _log_batch(self, batch_idx, batch, mode="train"):
        pass
