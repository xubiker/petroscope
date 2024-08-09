from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class ExMetric:
    """
    Base class for metrics that can be reduced to a single value.

    Attributes:
        value (float): The value of the metric.

    Methods:
        reduce(metrics: Iterable["ExMetric"]) -> "ExMetric": Reduces a list of metrics to a single metric.

        __str__(self) -> str: Returns a string representation of the metric value.
    """

    @property
    def value(self) -> float:
        pass

    @staticmethod
    def reduce(metrics: Iterable["ExMetric"]) -> "ExMetric":
        pass

    def __str__(self) -> str:
        return f"{self.value:.3f}"


class ExAcc(ExMetric):
    """
    Class representing the accuracy metric.

    Attributes:
        correct (int): The number of correct predictions.
        total (int): The total number of predictions.

    Methods:
        value(self) -> float: Returns the accuracy value.
        reduce(metrics: Iterable["ExAcc"]) -> "ExAcc": Reduces a list of ExAcc metrics to a single ExAcc metric.
    """

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
    """
    Class representing the intersection over union (IoU) metric.

    Attributes:
        intersection (float): The number of pixels that are both in the predicted and ground truth.
        union (float): The number of pixels that are in either the predicted or the ground truth.
        smooth (float): A smoothing factor to avoid division by zero.

    Methods:
        value(self) -> float: Returns the IoU value.
        reduce(metrics: Iterable["ExIoU"]) -> "ExIoU": Reduces a list of ExIoU metrics to a single ExIoU metric.
    """

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
    """
    Class representing the set of segmentation metrics (iou_soft, iou, acc) for a set of classes.

    Attributes:
        iou_soft (dict[str, ExIoU]): A dictionary mapping class labels to the soft IoU metric.
        iou (dict[str, ExIoU]): A dictionary mapping class labels to the IoU metric.
        acc (ExAcc): The accuracy metric.

    Methods:
        mean_iou(self) -> float: Calculates the mean intersection over union (IoU) of all classes.
        mean_iou_soft(self) -> float: Calculates the soft mean intersection over union (IoU) of all classes.
        reduce(results: Iterable["SegmMetrics"]) -> "SegmMetrics": Reduces a list of SegmMetrics to a single SegmMetrics.
        __str__(self) -> str: Returns a string representation of the SegmMetrics object.
    """

    iou_soft: dict[str, ExIoU]
    iou: dict[str, ExIoU]
    acc: ExAcc

    @property
    def mean_iou(self) -> float:
        """
        Calculates the mean intersection over union (IoU) of all classes.

        Returns:
            float: The mean IoU value.
        """
        return sum(i.value for i in self.iou.values()) / len(self.iou)

    @property
    def mean_iou_soft(self) -> float:
        """
        Calculates the soft mean intersection over union (IoU) of all classes.

        Returns:
            float: The mean IoU value.
        """
        return sum(i.value for i in self.iou_soft.values()) / len(
            self.iou_soft
        )

    @staticmethod
    def reduce(results: Iterable["SegmMetrics"]) -> "SegmMetrics":
        """
        Reduces a list of SegmMetrics to a single SegmMetrics.

        This function takes a list of SegmMetrics as input, reduces each metric (iou_soft, iou, acc)
        by calling their respective reduce methods, and returns a new SegmMetrics object with the
        reduced values.

        Args:
            results (Iterable[SegmMetrics]): A list of SegmMetrics to be reduced.

        Returns:
            SegmMetrics: A new SegmMetrics object with the reduced values.
        """
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
    """
    Computes the Intersection over Union (IoU) metric for two tensors using TensorFlow.

    Args:
        y_true (tensor): The ground truth tensor.
        y_pred (tensor): The predicted tensor.
        smooth (float, optional): A smoothing factor to avoid division by zero. Defaults to 1.0.

    Returns:
        float: The IoU value.
    """
    from tensorflow.keras import backend as K

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou(y_true: np.ndarray, y_pred: np.ndarray, smooth=1e-3) -> ExIoU:
    """
    Calculate the Intersection over Union (IoU) metric between the ground truth and the predicted mask.

    Args:
        y_true (numpy.ndarray): The ground truth mask.
        y_pred (numpy.ndarray): The predicted mask.
        smooth (float, optional): A smoothing factor to avoid division by zero. Defaults to 1e-3.

    Returns:
        ExIoU: An instance of the ExIoU class representing the IoU metric.

    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return ExIoU(intersection=intersection, union=union, smooth=smooth)


def to_hard(pred: np.ndarray) -> np.ndarray:
    """
    Convert a soft prediction tensor to a hard prediction tensor.

    Args:
        pred (np.ndarray): The soft prediction tensor.

    Returns:
        np.ndarray: The hard prediction tensor.

    """
    n_cl = pred.shape[-1]
    c = np.argmax(pred, axis=-1)
    return np.eye(n_cl)[c]  # same as to_categorical


def iou_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_labels: dict[int, str],
    smooth=1e-3,
) -> dict[str, ExIoU]:
    """
    Calculate the Intersection over Union (IoU) metric per class for the given ground truth and predicted masks.

    Args:
        y_true (numpy.ndarray): The ground truth mask.
        y_pred (numpy.ndarray): The predicted mask.
        idx_to_labels (dict[int, str]): A dictionary mapping class indices to their corresponding labels.
        smooth (float, optional): A smoothing factor to avoid division by zero. Defaults to 1e-3.

    Returns:
        dict[str, ExIoU]: A dictionary mapping class labels to their corresponding IoU values.

    """
    iou_vals = dict()
    n_cl = y_pred.shape[-1]
    for i in range(n_cl):
        iou_vals[idx_to_labels[i]] = iou(
            y_true[..., i], y_pred[..., i], smooth=smooth
        )
    return iou_vals


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> ExAcc:
    """
    Calculate the accuracy between the ground truth and the predicted labels.

    Parameters:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        ExAcc: An instance of the ExAcc class representing the accuracy metric.
    """
    y_pred_a = np.argmax(y_pred, axis=-1)
    y_true_a = np.argmax(y_true, axis=-1)
    correct = np.sum(y_pred_a == y_true_a)
    return ExAcc(correct=correct, total=y_pred_a.size)
