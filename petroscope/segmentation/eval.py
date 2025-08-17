"""
Segmentation model evaluation with metrics, visualization, and logging.

Provides classes for evaluating segmentation models with support for both
full and void-aware evaluation.
"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from petroscope.segmentation.classes import ClassSet
from petroscope.segmentation.loggers import DetailedTestLogger
from petroscope.segmentation.metrics import (
    SegmMetrics,
    acc,
    iou_per_class,
    to_hard,
)
from petroscope.segmentation.utils import (
    load_image,
    load_mask,
    to_categorical,
    void_borders,
)
from petroscope.segmentation.vis import SegmVisualizer


class SegmEvaluator:
    """
    Core evaluator for segmentation model performance.

    Evaluates metrics on individual images and accumulates results for
    dataset-level evaluation. Supports both soft and hard predictions
    with optional void mask handling.

    Attributes
    ----------
    code_to_label : dict
        Class code to label mappings.
    buffer : list[SegmMetrics]
        Buffer for individual image metrics.
    """

    def __init__(self, code_to_label) -> None:
        """
        Initialize evaluator with class label mappings.

        Parameters
        ----------
        code_to_label : dict
            Class codes to string labels mapping.
        """
        self.code_to_label = code_to_label
        self.buffer: list[SegmMetrics] = []

    def evaluate(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        void_mask: np.ndarray = None,
        add_to_buffer=True,
    ) -> SegmMetrics:
        """
        Evaluate segmentation model on a single image.

        Parameters
        ----------
        pred : np.ndarray
            Predicted mask, 2D (class indices) or 3D (probabilities).
        gt : np.ndarray
            Ground truth one-hot encoded mask (H, W, C).
        void_mask : np.ndarray, optional
            Binary mask of invalid pixels to exclude.
        add_to_buffer : bool, default True
            Whether to add metrics to buffer for aggregation.

        Returns
        -------
        SegmMetrics
            Computed metrics including IoU and accuracy.
        """
        assert pred.ndim >= 2, "Prediction must be at least 2D"

        # Convert flat prediction to categorical if needed
        if pred.ndim == 2:
            pred = to_categorical(pred, gt.shape[-1])

        # Create hard (argmax) version of the prediction
        pred_hard = to_hard(pred)

        # Apply void mask if provided
        if void_mask is not None:
            assert (
                void_mask.shape[:2] == gt.shape[:2]
            ), "Void mask spatial dimensions must match ground truth"
            # Expand void mask to match channel dimension if needed
            void = (
                np.repeat(void_mask[..., np.newaxis], gt.shape[-1], axis=-1)
                if void_mask.ndim == 2
                else void_mask
            )
            # Zero out void regions
            pred *= void
            pred_hard *= void
            gt *= void

        # Calculate per-class IoU
        iou_class_soft = iou_per_class(gt, pred, self.code_to_label)
        iou_class_hard = iou_per_class(gt, pred_hard, self.code_to_label)

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
        Aggregate buffered metrics into dataset-level statistics.

        Returns
        -------
        SegmMetrics
            Aggregated metrics for the entire dataset.
        """
        ds_metrics = SegmMetrics.reduce(self.buffer)
        self.buffer.clear()
        return ds_metrics


class SegmDetailedTester:
    """
    Advanced segmentation tester with visualization and logging.

    Evaluates models on datasets with visualization generation and detailed
    per-image logging. Provides both full and void-aware evaluation.

    Attributes
    ----------
    out_dir : Path
        Output directory for results and visualizations.
    classes : ClassSet
        Class definitions for the segmentation task.
    void_w : int
        Border width for void region generation.
    void_pad : int
        Additional padding for void regions.
    vis_segmentation : bool
        Flag to enable visualization output.
    detailed_logger : DetailedTestLogger, optional
        Logger for detailed per-image metrics.
    eval_full : SegmEvaluator
        Evaluator for full image metrics.
    eval_void : SegmEvaluator
        Evaluator for void-aware metrics.
    """

    def __init__(
        self,
        out_dir: Path,
        classes: ClassSet,
        max_classes: int,
        void_pad: int = 0,
        void_border_width: int = 0,
        void_rare_classes: list[int] | None = None,
        vis_segmentation: bool = True,
        detailed_logger: DetailedTestLogger = None,
    ):
        """
        Initialize the detailed tester.

        Parameters
        ----------
        out_dir : Path
            Directory for test results and visualizations.
        classes : ClassSet
            Class definitions with label mappings.
        void_pad : int, default 0
            Additional padding for void regions.
        void_border_width : int, default 0
            Width of border regions to mark as void.
        void_rare_classes : list[int], optional
            List of class codes to void during testing.
        vis_segmentation : bool, default True
            Whether to generate visualizations.
        detailed_logger : DetailedTestLogger, optional
            Logger for detailed per-image metrics.
        """
        self.vis_segmentation = vis_segmentation
        self.out_dir = out_dir
        self.classes = classes
        self.max_classes = max_classes
        self.void_w = void_border_width
        self.void_pad = void_pad
        self.void_rare_classes = void_rare_classes or []
        self.eval_full = SegmEvaluator(code_to_label=classes.code_to_label)
        self.eval_void = SegmEvaluator(code_to_label=classes.code_to_label)
        self.detailed_logger = detailed_logger

    def _apply_void_rare_classes(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply void rare classes to ground truth mask.

        Parameters
        ----------
        mask : np.ndarray
            Ground truth mask (one-hot or categorical).

        Returns
        -------
        np.ndarray
            Mask with rare classes voided.
        """
        if not self.void_rare_classes:
            return mask

        mask_copy = mask.copy()
        # Handle both categorical and one-hot masks
        if mask.ndim == 2:  # Categorical mask
            for class_code in self.void_rare_classes:
                mask_copy[mask_copy == class_code] = 255
        else:  # One-hot mask
            for class_code in self.void_rare_classes:
                if class_code < mask.shape[-1]:
                    mask_copy[..., class_code] = 0

        return mask_copy

    def _visualize(
        self,
        img: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        void_mask: np.ndarray | None,
        out_dir: Path,
        img_name: str,
    ) -> None:
        """
        Generate and save segmentation visualization composite.

        Parameters
        ----------
        img : np.ndarray
            Original input image (H, W, 3).
        gt_mask : np.ndarray
            Ground truth mask, categorical or one-hot.
        pred_mask : np.ndarray
            Predicted mask, categorical or probabilistic.
        void_mask : np.ndarray or None
            Binary mask of void regions.
        out_dir : Path
            Directory for visualization output.
        img_name : str
            Base name for output file.
        """

        # Generate composite visualization
        composite_vis = SegmVisualizer.vis_test(
            source_bgr=img[:, :, ::-1],  # Convert RGB to BGR
            mask_gt=gt_mask,
            mask_pred=pred_mask,
            classes=self.classes,
            void_mask=void_mask,
        )

        # Save with high quality
        cv2.imwrite(
            str(out_dir / f"{img_name}_composite.jpg"),
            composite_vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

    def _log_image_metrics(
        self,
        epoch: int,
        name: str,
        metrics_full: SegmMetrics,
        metrics_void: SegmMetrics,
    ) -> None:
        """
        Log detailed metrics for a single image.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        name : str
            Image identifier.
        metrics_full : SegmMetrics
            Full image metrics.
        metrics_void : SegmMetrics
            Void-aware metrics.
        """
        if self.detailed_logger:
            self.detailed_logger.log_image_metrics(
                epoch=epoch,
                image_name=name,
                metrics=metrics_full,
                void=False,
            )
            self.detailed_logger.log_image_metrics(
                epoch=epoch,
                image_name=name,
                metrics=metrics_void,
                void=True,
            )

    def test_on_set(
        self,
        img_mask_paths: Iterable[tuple[Path, Path]],
        predict_func,
        epoch: int = 0,
    ) -> tuple[SegmMetrics, SegmMetrics]:
        """
        Evaluate segmentation model on a complete dataset.

        Parameters
        ----------
        img_mask_paths : Iterable[tuple[Path, Path]]
            Iterable of (image_path, mask_path) tuples.
        predict_func : callable
            Function that takes an image and returns prediction.
        epoch : int, default 0
            Current epoch number for output organization.

        Returns
        -------
        tuple[SegmMetrics, SegmMetrics]
            Tuple of (full_metrics, void_metrics).
        """
        sub_dir = self.out_dir / f"epoch_{epoch}"
        sub_dir.mkdir(exist_ok=True, parents=True)

        for img_mask_path in tqdm(img_mask_paths, "testing"):
            name = img_mask_path[0].stem
            img = load_image(img_mask_path[0], normalize=False)
            mask = load_mask(
                img_mask_path[1],
                max_classes=self.max_classes,
                one_hot=True,
            )

            # Apply void rare classes to ground truth
            mask = self._apply_void_rare_classes(mask)

            pred = predict_func(img)
            void = void_borders(
                mask, border_width=self.void_w, pad=self.void_pad
            )

            # Evaluate with both full and void-aware metrics
            metrics_full = self.eval_full.evaluate(pred, gt=mask)
            metrics_void = self.eval_void.evaluate(
                pred, gt=mask, void_mask=void
            )

            # Log per-image metrics
            self._log_image_metrics(epoch, name, metrics_full, metrics_void)

            # Generate visualization if enabled
            if self.vis_segmentation:
                self._visualize(img, mask, pred, void, sub_dir, f"img_{name}")

        # Aggregate metrics for the dataset
        metrics_set = self.eval_full.flush()
        metrics_void_set = self.eval_void.flush()

        return metrics_set, metrics_void_set
