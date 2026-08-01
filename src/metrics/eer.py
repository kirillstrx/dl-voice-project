import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size

    all_scores = np.concatenate((target_scores, nontarget_scores))

    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")

    labels = labels[indices]

    target_trial_sums = np.cumsum(labels)

    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - target_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), target_trial_sums / target_scores.size))

    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))

    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)

    absolute_differences = np.abs(frr - far)
    minimum_index = np.argmin(absolute_differences)

    eer = np.mean((frr[minimum_index], far[minimum_index]))

    return (eer, thresholds[minimum_index])


class AccuracyMetric(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        predictions = logits.argmax(dim=-1)

        accuracy = (predictions == labels).float().mean()

        return accuracy.item()


class EERMetric(BaseMetric):

    requires_full_dataset = True

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        scores = logits[:, 1] - logits[:, 0]

        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        eer, _ = compute_eer(bonafide_scores, spoof_scores)

        return 100.0 * float(eer)
