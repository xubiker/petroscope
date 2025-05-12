from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from petroscope.segmentation.classes import ClassSet
from petroscope.segmentation.metrics import SegmMetrics


class SegmVisualizer:
    """
    This class provides methods for visualizing segmentation masks.

    """

    @staticmethod
    def colorize_mask(
        mask: np.ndarray,
        values_to_colors_rgb: dict[int, tuple[int, int, int]],
    ) -> np.ndarray:
        """
        Colorize a segmentation mask based on the provided class indices
        to colors mapping. Colors are provided in RGB format.

        Args:
            mask: The input segmentation mask to colorize
            values_to_colors: A dictionary mapping mask values to
                corresponding RGB colors

        Returns:
            Colorized mask as numpy array in RGB format
        """
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]

        res = np.zeros(mask.shape + (3,), dtype=np.uint8)
        # set colors for each class
        for code, color in values_to_colors_rgb.items():
            res[mask == code] = color[::-1]  # Convert RGB to BGR for OpenCV
        return res

    @staticmethod
    def _create_legend(
        classes: list[tuple[str, tuple[int, int, int]]],
        width: int,
        font_scale: float = 2.5,
        line_thickness: int = 3,
        box_size: int = 50,
        element_padding: int = 50,
        box_text_padding: int = 20,
    ) -> np.ndarray:
        """
        Create a legend image showing class colors and labels using OpenCV.
        The height of the legend is determined automatically.

        Args:
            classes: List of (label, color) tuples. Colors should be in RGB
                    format (tuple).
            width: Width of the legend image
            font_scale: Scale of the font
            line_thickness: Thickness of the text lines
            box_size: Size of the color boxes
            element_padding: Padding between elements
            box_text_padding: Padding between box and text

        Returns:
            Legend image as numpy array in BGR (OpenCV format)
        """

        # Create a blank image for the  legend if there are no classes
        if not classes:
            return np.ones((element_padding, width, 3), dtype=np.uint8) * 255

        # Calculate each item's width based on its text length
        item_widths = []
        text_heights = []
        for label, _ in classes:
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            # Item width = box width + padding + text width
            item_widths.append(box_size + box_text_padding + text_size[0])
            text_heights.append(text_size[1])

        # Determine how many items to place in each row
        # Start with all items in a single row
        rows = []
        current_row = []
        cur_row_width = 0

        for i, item_width in enumerate(item_widths):
            # If adding this item would exceed the width,
            # start a new row - unless it's the first item in the row
            if (
                cur_row_width + element_padding + item_width
                > width - 2 * element_padding
                and current_row
            ):
                rows.append(current_row)
                current_row = [i]
                cur_row_width = item_width
            else:
                current_row.append(i)
                # Add width of item plus padding between items
                if current_row:
                    cur_row_width += element_padding + item_width
                else:
                    cur_row_width = item_width

        # Add the last row if it's not empty
        if current_row:
            rows.append(current_row)

        # Calculate the height needed for the legend
        row_height = max(box_size, max(text_heights))
        required_height = (
            (len(rows) * row_height)
            + ((len(rows) - 1) * element_padding)
            + 2 * element_padding
        )

        # Create a blank image for the legend with the adjusted height
        legend_image = (
            np.ones((required_height, width, 3), dtype=np.uint8) * 255
        )

        for row_idx, row in enumerate(rows):
            # Calculate the total width of items in this row
            row_width = sum(item_widths[idx] for idx in row)
            # Add padding between items (number of items - 1)
            row_width += element_padding * (len(row) - 1)

            # Center the row horizontally
            start_x = (width - row_width) // 2

            # Vertical position for this row
            # (account for vertical padding between rows)
            y_pos = element_padding + row_idx * (row_height + element_padding)

            # Reset x position for each row
            x_pos = start_x

            for item_idx in row:
                label, color = classes[item_idx]
                # Draw color box
                cv2.rectangle(
                    legend_image,
                    (x_pos, y_pos),
                    (x_pos + box_size, y_pos + box_size),
                    color[::-1],  # Convert RGB to BGR for OpenCV
                    -1,  # Filled rectangle
                )

                # Get text size again for vertical centering
                text_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
                )

                # Center text vertically relative to the color box
                text_y = y_pos + (box_size + text_size[1]) // 2

                # Draw text
                cv2.putText(
                    legend_image,
                    label,
                    (x_pos + box_size + box_text_padding, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    line_thickness,
                    cv2.LINE_AA,
                )

                # Move x position for the next item
                x_pos += item_widths[item_idx] + element_padding

        return legend_image

    @staticmethod
    def _create_header(
        text: str,
        width: int,
        padding: int = 50,
        font_scale: float = 2.5,
        line_thickness: int = 3,
    ) -> np.ndarray:
        """
        Create a header image with text using OpenCV.
        The header's height is determined automatically based on the text size.

        Args:
            text: The header text
            width: Width of the header image
            font_scale: Scale of the font
            line_thickness: Thickness of the text lines

        Returns:
            Header image as numpy array in BGR (OpenCV format)
        """

        if not text:
            return np.zeros((padding, width, 3), dtype=np.uint8)

        # Get text size
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )

        # Calculate the height of the header
        height = text_size[1] + 2 * padding

        # Create a blank image for the header
        header_image = np.full((height, width, 3), 255, dtype=np.uint8)

        # Calculate center position, accounting for padding
        # Ensure the text remains within the padded area
        pos_x = (width - text_size[0]) // 2
        pos_y = padding + text_size[1]

        # Draw text
        cv2.putText(
            header_image,
            text,
            (pos_x, pos_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            line_thickness,
            cv2.LINE_AA,
        )

        return header_image

    @staticmethod
    def highlight_mask_np(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Highlight areas in an image based on a binary mask by dimming non-mask
        areas and brightening mask areas.

        Args:
            img: Input image in BGR format
            mask: Binary mask where positive values indicate areas to highlight

        Returns:
            Highlighted image in BGR format
        """
        if img.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "Image must be 3D (H, W, C) and mask must be 2D (H, W)"
            )
        mask = np.stack([mask, mask, mask], axis=-1)
        # Dim the non-highlighted areas
        img_dimmed = img.copy() * 0.1
        # Brighten the highlighted areas
        img_lighted = np.clip(img.copy() * 1.2, a_min=0, a_max=255)
        # Combine using the mask
        res = img_dimmed * (1 - mask) + (img_lighted - img_dimmed) * mask
        return res.astype(np.uint8)

    @staticmethod
    def compose(
        items: list[np.ndarray],
        legend_data: list[tuple[str, tuple[int, int, int]]] = None,
        header_data: str = None,
        padding=50,
    ) -> np.ndarray:
        """
        Compose multiple images horizontally with optional header and legend.
        All images should be in BGR (OpenCV format).

        Args:
            items: List of images to compose horizontally
            legend_data: List of (label, color) tuples for the legend
            header_data: Header text
            padding: Padding between images

        Returns:
            Composite image as numpy array in BGR (OpenCV format)
        """

        if len(items) == 0:
            raise ValueError("No images to compose")
        if not all(v.ndim == 3 for v in items):
            raise ValueError("All inputs must be 3D (H, W, C) arrays")
        if not all(v.dtype == np.uint8 for v in items):
            raise ValueError("All inputs must be of type uint8")
        if not all(v.shape[:2] == items[0].shape[:2] for v in items):
            raise ValueError("All inputs must have the same height and width")

        gap_v = np.full(
            [items[0].shape[0], padding, 3], (255, 255, 255), dtype=np.uint8
        )

        # insert gap between visualizations
        items = [
            item
            for pair in zip(items, [gap_v] * (len(items) - 1))
            for item in pair
        ] + [items[-1]]

        v = np.pad(
            np.concatenate(items, axis=1),
            (
                (
                    padding if header_data is None else 0,
                    padding if legend_data is None else 0,
                ),
                (padding, padding),
                (0, 0),
            ),
            constant_values=255,
        )

        if legend_data is not None:
            # create the legend
            legend_image = SegmVisualizer._create_legend(
                legend_data,
                width=v.shape[1],
            )
            v = np.concatenate((v, legend_image), axis=0)

        if header_data is not None:
            # create header
            header_image = SegmVisualizer._create_header(
                header_data,
                width=v.shape[1],
            )
            v = np.concatenate((header_image, v), axis=0)

        return v

    @staticmethod
    def _vis_src_with_mask(
        source: np.ndarray,
        mask: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool,
        overlay_alpha: float,
        show_legend: bool,
        header_data: str,
    ) -> np.ndarray:

        if source.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "Source must be 3D (H, W, C) and mask must be 2D (H, W)"
            )
        if source.dtype != np.uint8 or mask.dtype != np.uint8:
            raise ValueError(
                "Source must be of type uint8 and mask must be of type uint8"
            )

        if source.shape[:2] != mask.shape:
            raise ValueError(
                "Source and mask must have the same height and width"
            )
        if not 0 <= overlay_alpha <= 1:
            raise ValueError("Overlay alpha must be between 0 and 1")

        mask_colored = SegmVisualizer.colorize_mask(
            mask,
            classes.colors_map(squeezed=classes_squeezed),
        )

        overlay = cv2.addWeighted(
            src1=source,
            alpha=overlay_alpha,
            src2=mask_colored,
            beta=1 - overlay_alpha,
            gamma=0,
        )

        codes = np.unique(mask).tolist()
        if classes_squeezed:
            codes = [classes.idx_to_code[i] for i in codes]

        legend_items = (
            [
                (f"{cl.label} ({cl.name})", cl.color_rgb)
                for cl in classes.classes
                if cl.code in codes
            ]
            if show_legend
            else None
        )

        return SegmVisualizer.compose(
            [source, overlay, mask_colored],
            legend_data=legend_items,
            header_data=header_data,
        )

    @staticmethod
    def vis_annotation(
        source_bgr: np.ndarray,
        mask: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool = False,
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Visualize a segmentation annotation on top of the source image.

        Args:
            source_bgr: The source image (BGR format)
            mask: The segmentation mask (2D array of class indices)
            classes: The set of segmentation classes
            classes_squeezed: If True, the class indices in the mask represent
                the classes directly; otherwise, the class indices represent
                indices into the classes array
            overlay_alpha: The alpha value for the overlay (between 0 and 1)
            show_legend: If True, show a legend with class names and colors

        Returns:
            The visualization as a numpy array in BGR format
        """
        return SegmVisualizer._vis_src_with_mask(
            source=source_bgr,
            mask=mask,
            classes=classes,
            classes_squeezed=classes_squeezed,
            overlay_alpha=overlay_alpha,
            show_legend=show_legend,
            header_data="source   |   overlay   |   annotation",
        )

    @staticmethod
    def vis_prediction(
        source_bgr: np.ndarray,
        pred: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool = False,
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        return SegmVisualizer._vis_src_with_mask(
            source=source_bgr,
            mask=pred,
            classes=classes,
            classes_squeezed=classes_squeezed,
            overlay_alpha=overlay_alpha,
            show_legend=show_legend,
            header_data="source   |   overlay   |   prediction",
        )

    @staticmethod
    def vis_test(
        source_bgr: np.ndarray,
        mask_gt: np.ndarray,
        mask_pred: np.ndarray,
        classes: ClassSet,
        void_mask: np.ndarray = None,
        mask_gt_squeezed: bool = False,
        mask_pred_squeezed: bool = False,
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Visualizes the comparison between ground truth and predicted
        segmentation masks.

        Args:
            source_bgr (np.ndarray): The source image in BGR format.

            mask_gt (np.ndarray): The ground truth segmentation mask.

            mask_pred (np.ndarray): The predicted segmentation mask.

            classes (ClassSet): The set of classes.

            void_mask (np.ndarray, optional): A mask indicating void areas.
            Defaults to None.

            mask_gt_squeezed (bool, optional): Whether the ground truth mask
            is squeezed. Defaults to False.

            mask_pred_squeezed (bool, optional): Whether the predicted mask
            is squeezed. Defaults to False.

            show_legend (bool, optional): Whether to show the legend in the
            visualization. Defaults to True.

        Returns:
            np.ndarray: The composite visualization image.
        """
        pred_colored = SegmVisualizer.colorize_mask(
            mask_pred,
            classes.colors_map(squeezed=mask_pred_squeezed),
        )

        gt_colored = SegmVisualizer.colorize_mask(
            mask_gt,
            classes.colors_map(squeezed=mask_gt_squeezed),
        )

        correct = (mask_gt == mask_pred).astype(np.uint8)
        if void_mask is not None:
            correct[void_mask == 0] = 255

        correct_colored = SegmVisualizer.colorize_mask(
            correct,
            values_to_colors_rgb={
                0: (244, 67, 54),
                1: (76, 175, 80),
                255: (0, 0, 255),  # void color
            },
        )

        error_overlay = SegmVisualizer.highlight_mask_np(
            source_bgr, (mask_gt != mask_pred).astype(np.uint8) * void_mask
        )

        codes_pred = np.unique(mask_pred).tolist()
        codes_gt = np.unique(mask_gt).tolist()

        if mask_pred_squeezed:
            codes_pred = [classes.idx_to_code[i] for i in codes_pred]
            codes_gt = [classes.idx_to_code[i] for i in codes_gt]

        codes = sorted(list(set(codes_gt) | set(codes_pred)))

        legend_items = (
            [
                (f"{cl.label} ({cl.name})", cl.color_rgb)
                for cl in classes.classes
                if cl.code in codes
            ]
            if show_legend
            else None
        )

        return SegmVisualizer.compose(
            [
                source_bgr,
                gt_colored,
                pred_colored,
                correct_colored,
                error_overlay,
            ],
            legend_data=legend_items,
            header_data=(
                "source   |   ground truth   |   prediction   "
                "|   error map   |   error highlight"
            ),
        )


class Plotter:
    """
    This class contains static methods for plotting various metrics
    and learning rate schedules.
    """

    @staticmethod
    def plot_single_class_metric(
        out_dir: Path,
        metric_name: str,
        values: Iterable[float],
        name_suffix: str = "",
    ):
        """
        Plots a single class metric over epochs.

        Args:
            out_dir (Path): The output directory to save the plot.
            metric_name (str): The name of the metric to plot.
            values (Iterable[float]): The values of the metric over epochs.
            name_suffix (str, optional): A suffix to append to the metric name.
            Defaults to "".

        Returns:
            None
        """
        epochs = len(values)
        fig = plt.figure(figsize=(12, 6))
        # ax = plt.axes()
        # ax.set_facecolor('white')
        x = [x + 1 for x in range(epochs)]
        y = [values[i] for i in range(epochs)]
        plt.plot(x, y)
        # plt.suptitle(f'{metric_name} over epochs', fontsize=20)
        plt.ylabel(f"{metric_name}{name_suffix}", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        fig.savefig(out_dir / f"{metric_name}{name_suffix}.png")

    @staticmethod
    def plot_multi_class_metric(
        out_dir: Path,
        metric_name,
        data: dict[str, Iterable[float]],
        colors: dict[str, tuple[float, float, float]],
        name_suffix: str = "",
    ):
        epochs = len(list(data.values())[0])
        fig = plt.figure(figsize=(12, 6))
        # ax = plt.axes()
        # ax.set_facecolor('white')
        for cl, vals in data.items():
            x = [x + 1 for x in range(epochs)]
            y = [vals[i] for i in range(epochs)]
            plt.plot(x, y, color=colors[cl])
        # plt.suptitle(f'{metric_name} per class over epochs', fontsize=20)
        plt.ylabel(f"{metric_name}{name_suffix}", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        plt.legend(
            [cl_str for cl_str in data], loc="center right", fontsize=15
        )
        fig.savefig(out_dir / f"{metric_name}{name_suffix}.png")

    @staticmethod
    def plot_segm_metrics(
        metrics: Iterable[SegmMetrics],
        out_dir: Path,
        colors: dict[str, tuple[float, float, float]],
        name_suffix: str = "",
    ):
        """
        Plots the segmentation metrics for a given set of SegmMetrics objects.

        Args:
            metrics (Iterable[SegmMetrics]): An iterable of SegmMetrics
            objects containing the metrics to be plotted.

            out_dir (Path): The output directory where the plots will be saved.

            colors (dict[str, tuple[float, float, float]]): A dictionary
            mapping class labels to their RGB colors.

            name_suffix (str, optional): A suffix to be added to the plot
            filenames. Defaults to "".

        Returns:
            None
        """
        labels = metrics[0].iou.keys()

        # transform metrics data to plot data
        single_class_plot_data = {
            "acc": [m.acc.value for m in metrics],
            "mean_iou_soft": [m.mean_iou_soft for m in metrics],
            "mean_iou": [m.mean_iou for m in metrics],
        }
        multi_class_plot_data = {
            "iou_soft": {
                label: [m.iou_soft[label].value for m in metrics]
                for label in labels
            },
            "iou": {
                label: [m.iou[label].value for m in metrics]
                for label in labels
            },
        }

        # perform plotting
        for metric_name, data in single_class_plot_data.items():
            Plotter.plot_single_class_metric(
                out_dir, metric_name, data, name_suffix=name_suffix
            )
        for metric_name, data in multi_class_plot_data.items():
            Plotter.plot_multi_class_metric(
                out_dir,
                metric_name,
                data,
                colors=colors,
                name_suffix=name_suffix,
            )

    @staticmethod
    def plot_lrs(lrs: list, output_path: Path):
        """
        Plots the learning rate schedule and saves it as an image.

        Args:
            lrs (list): A list of learning rates.
            output_path (Path): The path where the image will be saved.
        """
        plt.style.use("ggplot")
        fig = plt.figure()
        plt.plot([i + 1 for i in range(0, len(lrs))], lrs)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        fig.savefig(output_path / "lrs.jpg")
        plt.close()
