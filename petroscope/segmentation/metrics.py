from dataclasses import dataclass
from typing import Iterable
import numpy as np


class ExMetric:

    @property
    def value(self) -> float:
        pass

    @staticmethod
    def reduce(metrics: Iterable["ExMetric"]) -> "ExMetric":
        pass

    def __str__(self) -> str:
        return f"{self.value:.3f}"


class ExAcc(ExMetric):
    def __init__(self, correct: int, total: int) -> None:
        self.correct = correct
        self.total = total

    @property
    def value(self) -> float:
        return self.correct / self.total

    @staticmethod
    def reduce(metrics: Iterable["ExAcc"]) -> "ExAcc":
        correct = sum(a.correct for a in metrics)
        total = sum(a.total for a in metrics)
        return ExAcc(correct=correct, total=total)


class ExIoU(ExMetric):
    def __init__(
        self, intersection: float, union: float, smooth: float = 1e-3
    ) -> None:
        self.intersection = intersection
        self.union = union
        self.smooth = smooth

    @property
    def value(self) -> float:
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    @staticmethod
    def reduce(metrics: Iterable["ExIoU"]) -> "ExIoU":
        intersection = sum(a.intersection for a in metrics)
        union = sum(a.union for a in metrics)
        return ExIoU(
            intersection=intersection,
            union=union,
        )


@dataclass
class SegmMetrics:
    iou_soft: dict[str, ExIoU]
    iou: dict[str, ExIoU]
    acc: ExAcc

    @property
    def mean_iou(self) -> float:
        return sum(i.value for i in self.iou.values()) / len(self.iou)

    @property
    def mean_iou_soft(self) -> float:
        return sum(i.value for i in self.iou_soft.values()) / len(
            self.iou_soft
        )

    @staticmethod
    def reduce(results: Iterable["SegmMetrics"]) -> "SegmMetrics":
        iou_soft = {
            cl: ExIoU.reduce([r.iou_soft[cl] for r in results])
            for cl in results[0].iou_soft
        }
        iou = {
            cl: ExIoU.reduce([r.iou[cl] for r in results])
            for cl in results[0].iou
        }

        acc = ExAcc.reduce([r.acc for r in results])

        return SegmMetrics(
            iou_soft=iou_soft,
            iou=iou,
            acc=acc,
        )

    def __str__(self) -> str:
        iou_cl_str = "".join(
            (
                f"\t\t {cl}: {self.iou[cl].value:.4f} "
                f"[{self.iou_soft[cl].value:.4f}]\n"
            )
            for cl in sorted(self.iou.keys())
        )
        s = (
            f"\t iou [soft]:\n"
            f"{iou_cl_str}"
            f"\t mean iou [soft]: {self.mean_iou:.4f} [{self.mean_iou_soft:.4f}]\n"
            f"\t acc: {self.acc.value:.4f}\n"
        )
        return s


def iou_tf(y_true, y_pred, smooth=1.0):
    from tensorflow.keras import backend as K

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou(y_true, y_pred, smooth=1e-3) -> ExIoU:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return ExIoU(intersection=intersection, union=union, smooth=smooth)


def to_hard(pred: np.ndarray) -> np.ndarray:
    n_cl = pred.shape[-1]
    c = np.argmax(pred, axis=-1)
    return np.eye(n_cl)[c]  # same as to_categorical


def iou_per_class(
    y_true, y_pred, codes_to_labels: dict[int, str], smooth=1e-3
) -> dict[str, ExIoU]:
    iou_vals = dict()
    n_cl = y_pred.shape[-1]
    for i in range(n_cl):
        iou_vals[codes_to_labels[i]] = iou(
            y_true[..., i], y_pred[..., i], smooth=smooth
        )
    return iou_vals


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> ExAcc:
    y_pred_a = np.argmax(y_pred, axis=-1)
    y_true_a = np.argmax(y_true, axis=-1)
    correct = np.sum(y_pred_a == y_true_a)
    return ExAcc(correct=correct, total=y_pred_a.size)
