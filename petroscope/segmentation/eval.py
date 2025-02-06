from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from petroscope.segmentation.metrics import (
    SegmMetrics,
    acc,
    iou_per_class,
    to_hard,
)
from petroscope.segmentation.utils.data import (
    ClassSet,
    load_image,
    load_mask,
    to_categorical,
    void_borders,
)
from petroscope.segmentation.utils.vis import Plotter, SegmVisualizer


class SegmEvaluator:
    """
    Class for evaluating segmentation models on a set of images.

    This class provides methods to evaluate segmentation metrics.
    Metrics are calculated for each image and then
    reduced to a single metric for the whole dataset.

    Attributes:
    ----------
    idx_to_lbls (dict): A dictionary mapping class indices to their labels.
    buffer (list[SegmMetrics]): A buffer of metrics calculated for each image.

    Methods:
    --------
    evaluate(pred: np.ndarray, gt: np.ndarray, void_mask: np.ndarray = None,
    add_to_buffer=True) -> SegmMetrics:
        Evaluates segmentation model on a single image.

    flush() -> SegmMetrics:
        Reduces the buffer of metrics (calculated for each image) to a single
        metric.
    """

    def __init__(self, idx_to_labels) -> None:
        """
        Initializes the class with a dictionary mapping class indices to their
        labels.

        Parameters:
        -----------
        idx_to_labels (dict): A dictionary mapping class indices to their
        labels.
        """
        self.idx_to_lbls = idx_to_labels
        self.buffer: list[SegmMetrics] = []

    def evaluate(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        void_mask: np.ndarray = None,
        add_to_buffer=True,
    ) -> SegmMetrics:
        """
        Evaluates segmentation model on a single image.

        Parameters:
        -----------
        pred (np.ndarray): A predicted segmentation mask.
        gt (np.ndarray): A ground truth segmentation mask.
        void_mask (np.ndarray, optional): A mask of void pixels.
        add_to_buffer (bool, optional): Whether to add metrics to the buffer.

        Returns:
        --------
        img_metrics (SegmMetrics): Metrics for the image.
        """
        assert pred.ndim >= 2

        # check if prediction is flat, transform it to categorical
        if pred.ndim == 2:
            pred = to_categorical(pred, gt.shape[-1])

        # create a hard version of the prediction
        pred_hard = to_hard(pred)

        if void_mask is not None:
            assert void_mask.shape[:2] == gt.shape[:2]
            void = (
                np.repeat(void_mask[..., np.newaxis], gt.shape[-1], axis=-1)
                if void_mask.ndim == 2
                else void_mask
            )
            pred *= void
            pred_hard *= void
            gt *= void

        iou_class_soft = iou_per_class(gt, pred, self.idx_to_lbls)
        iou_class_hard = iou_per_class(gt, pred_hard, self.idx_to_lbls)

        img_metrics = SegmMetrics(
            iou_soft=iou_class_soft,
            iou=iou_class_hard,
            acc=acc(gt, pred_hard),
        )
        if add_to_buffer:
            self.buffer.append(img_metrics)
        return img_metrics

    def flush(self) -> SegmMetrics:
        """
        Reduces the buffer of metrics (calculated for each image) to a single
        metric.

        Returns:
        --------
        ds_metrics (SegmMetrics): Metrics for the dataset.
        """
        ds_metrics = SegmMetrics.reduce(self.buffer)
        self.buffer.clear()
        return ds_metrics


class SegmDetailedTester:
    """
    Class for detailed testing of segmentation models on a set of images.
    It provides methods to evaluate segmentation accuracy, IoU and visualize
    errors.
    Output is saved in a specified directory.

    Attributes:
    ----------
    out_dir : Path
        Path to directory where output will be saved.
    classes : ClassSet
        Classes used for this task.
    void_pad : int
        Padding for void pixels.
    void_border_width : int
        Border width for void pixels.
    vis_segmentation : bool
        If True, segmentation visualization is saved.
    vis_plots : bool
        If True, plots of metrics are saved.
    log : bool
        If True, metrics for each image as well as for the whole dataset
        are saved in a log file.

    Methods:
    --------
    test_on_set(img_mask_paths, predict_func, description, return_void=True)
        Evaluates segmentation model on a set of images.
    """

    def __init__(
        self,
        out_dir: Path,
        classes: ClassSet,
        void_pad: int = 0,
        void_border_width: int = 0,
        vis_segmentation: bool = True,
        vis_plots: bool = True,
        log: bool = True,
    ):
        self.vis_segmentation = vis_segmentation
        self.vis_plots = vis_plots
        self.log = log
        self.out_dir = out_dir
        self.classes = classes
        self.void_w = void_border_width
        self.void_pad = void_pad
        self.metrics_history: list[SegmMetrics] = []
        self.metrics_void_history: list[SegmMetrics] = []
        self.eval = SegmEvaluator(idx_to_labels=classes.idx_to_label)
        self.eval_void = SegmEvaluator(idx_to_labels=classes.idx_to_label)

    def _visualize(
        self,
        img: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        void_mask: np.ndarray | None,
        out_dir: Path,
        img_name: str,
    ) -> None:
        assert img is not None
        img = (img * 255).astype(np.uint8)

        pred = (
            pred_mask if pred_mask.ndim == 2 else np.argmax(pred_mask, axis=-1)
        ).astype(np.uint8)
        gt = (
            gt_mask if gt_mask.ndim == 2 else np.argmax(gt_mask, axis=-1)
        ).astype(np.uint8)

        # visualize prediction
        composite_vis = SegmVisualizer.vis_test(
            img,
            gt,
            pred,
            classes=self.classes,
            void_mask=void_mask,
            mask_gt_squeezed=True,
            mask_pred_squeezed=True,
        )
        composite_vis.save(out_dir / f"{img_name}_composite.jpg", quality=95)

    def _log(self, message: str, detailed: bool) -> None:
        log_name = "metrics_per_image.txt" if detailed else "metrics.txt"
        log = open(self.out_dir / log_name, "a+")
        log.write(message + "\n")
        log.close()

    def test_on_set(
        self,
        img_mask_paths: Iterable[tuple[Path, Path]],
        predict_func,
        description: str,
        return_void: bool = True,
    ) -> SegmMetrics:
        sub_dir = self.out_dir / description
        sub_dir.mkdir(exist_ok=True, parents=True)

        # iterate over all images in the set
        for img_mask_path in tqdm(img_mask_paths, "testing"):
            name = img_mask_path[0].stem
            img = load_image(img_mask_path[0], normalize=True)
            mask = load_mask(
                img_mask_path[1],
                classes=self.classes,
                one_hot=True,
            )
            pred = predict_func(img)
            void = void_borders(
                mask, border_width=self.void_w, pad=self.void_pad
            )
            # evaluate each image prediction (normal and void)
            metrics = self.eval.evaluate(pred, gt=mask)
            metrics_void = self.eval_void.evaluate(
                pred, gt=mask, void_mask=void
            )
            # log image metrics
            self._log(
                f"{description}, {name}:\n{metrics}\n"
                + f"{description}, {name} (void):\n{metrics_void}\n",
                detailed=True,
            )
            # visualize segmentation for image
            if self.vis_segmentation:
                self._visualize(img, mask, pred, void, sub_dir, f"img_{name}")

        # get total metrics for the set
        metrics_set = self.eval.flush()
        metrics_void_set = self.eval_void.flush()

        # save metrics to history (needed for plots)
        self.metrics_history.append(metrics_set)
        self.metrics_void_history.append(metrics_void_set)

        # log the total metrics
        log_str = (
            f"{description}, total:\n{metrics_set}\n"
            + f"{description}, total (void):\n{metrics_void_set}\n"
        )
        self._log(log_str, detailed=True)
        self._log(log_str, detailed=False)

        # draw plots
        if self.vis_plots:
            Plotter.plot_segm_metrics(
                self.metrics_history,
                self.out_dir,
                colors=self.classes.labels_to_colors_plt,
            )
            Plotter.plot_segm_metrics(
                self.metrics_void_history,
                self.out_dir,
                colors=self.classes.labels_to_colors_plt,
                name_suffix="_void",
            )

        # return resulted metrics
        if return_void:
            return metrics_set, metrics_void_set
        return metrics_set
