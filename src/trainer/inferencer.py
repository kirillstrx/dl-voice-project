import csv

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        return batch

    def _save_scores(self, audio_ids, logits):
        if self.save_path is None:
            return None

        scores = (
            logits[:, 1] - logits[:, 0]
        ).tolist()

        csv_name = self.cfg_trainer.get(
            "csv_name",
            "predictions.csv",
        )

        csv_path = self.save_path / csv_name

        with csv_path.open(
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(
                zip(audio_ids, scores)
            )

        return csv_path

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        all_logits = []
        all_labels = []
        all_audio_ids = []

        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
            desc=part,
            total=len(dataloader),
            ):
                batch = self.process_batch(batch)

                all_logits.append(
                    batch["logits"].detach().cpu()
                )
                all_labels.append(
                    batch["labels"].detach().cpu()
                )
                all_audio_ids.extend(
                    batch["audio_id"]
                )

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        for metric in self.metrics["inference"]:
            metric_value = metric(
                logits=logits,
                labels=labels,
            )

            self.evaluation_metrics.update(
                metric.name,
                metric_value,
            )

        self._save_scores(
            all_audio_ids,
            logits,
        )

        return self.evaluation_metrics.result()
    
