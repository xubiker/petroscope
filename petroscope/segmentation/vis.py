from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from petroscope.segmentation.metrics import SegmMetrics
from petroscope.segmentation.classes import ClassSet


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string to BGR format.

    Args:
        hex_color (str): Hex color string (e.g., "#FF5733")

    Returns:
        tuple[int, int, int]: Color in BGR format
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Return as BGR
    return b, g, r


class SegmVisualizer:
    """
    This class provides methods for visualizing segmentation masks.

    """

    @staticmethod
    def _load_as_array(a, dtype=np.uint8) -> np.ndarray:
        if isinstance(a, np.ndarray):
            return a.astype(dtype)
        elif isinstance(a, Image.Image):
            return np.array(a, dtype=dtype)
        elif isinstance(a, (str, Path)):
            img = cv2.imread(str(a))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {a}")
            return img.astype(dtype)
        else:
            raise TypeError(f"Unsupported type for loading: {type(a)}")

    @staticmethod
    def colorize_mask(
        mask: np.ndarray,
        values_to_colors: dict[int, tuple[int, int, int]],
    ) -> np.ndarray:
        """
        This function colorizes a segmentation mask based on the provided
        class indices to colors mapping.

        Args:
            mask (np.ndarray): The input segmentation mask to colorize.

            values_to_colors (dict[int, tuple[int, int, int]]): A dictionary
            mapping mask values to corresponding BGR colors.

        Returns:
            np.ndarray: The colorized segmentation mask as a 3D numpy array.
        """
        colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for code, color in values_to_colors.items():
            # OpenCV uses BGR ordering, so we need to reverse the color tuple
            bgr_color = (
                color
                if isinstance(color[0], (list, tuple))
                else (color[2], color[1], color[0])
            )
            colorized[mask == code, :] = bgr_color
        return colorized

    @staticmethod
    def overlay(
        mask: np.ndarray,
        overlay: np.ndarray | Image.Image | Path = None,
        alpha=0.6,
    ) -> np.ndarray:
        """
        Overlay a mask on an image or another mask.

        Args:
            mask (np.ndarray): The mask to be overlaid. It should have
            3 channels.

            overlay (np.ndarray | Image.Image | Path, optional): The image or
            mask to be overlaid on the mask. Defaults to None.

            alpha (float, optional): The transparency of the overlay. Defaults
            to 0.75.

        Returns:
            np.ndarray: The resulting image with the mask overlaid on
            the overlay.
        """

        assert mask.ndim == 3, "only 3-channel masks are supported"

        if overlay is not None:
            overlay = SegmVisualizer._load_as_array(overlay)
            assert overlay.shape[:2] == mask.shape[:2]
            assert overlay.ndim == 3
        else:
            overlay = mask.copy()

        return np.clip(
            (alpha * overlay + (1 - alpha) * mask).astype(np.uint8),
            0,
            255,
        )

    @staticmethod
    def _create_legend(
        classes: list[tuple[str, tuple[int, int, int] | str]],
        width: int,
        height: int,
        font_scale: float = 2.5,
        box_size: int = 50,
        padding: int = 50,
        line_thickness: int = 3,
        vertical_row_padding: int = 20,
    ) -> np.ndarray:
        """
        Create a legend image showing class colors and labels using OpenCV.

        Args:
            classes: List of (label, color) tuples. Colors can be in BGR
                    format (tuple) or hex string format like "#FFFFFF"
                    (RGB format).
            width: Width of the legend image
            height: Height of the legend image
            font_scale: Scale of the font
            box_size: Size of the color boxes
            padding: Padding between elements
            line_thickness: Thickness of the text lines
            vertical_row_padding: Additional padding between rows

        Returns:
            Legend image as numpy array in BGR format
        """
        if not classes:
            return np.ones((height, width, 3), dtype=np.uint8) * 255

        # Calculate each item's width based on its text length
        item_widths = []
        text_heights = []
        for label, _ in classes:
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            # Item width = box width + padding + text width
            item_widths.append(box_size + padding + text_size[0])
            text_heights.append(text_size[1])

        # Determine how many items to place in each row
        # Start with all items in a single row
        rows = []
        current_row = []
        current_row_width = 0

        for i, item_width in enumerate(item_widths):
            # If adding this item would exceed the width,
            # start a new row - unless it's the first item in the row
            if (
                current_row_width + item_width > width - 2 * padding
                and current_row
            ):
                rows.append(current_row)
                current_row = [i]
                current_row_width = item_width
            else:
                current_row.append(i)
                # Add width of item plus padding between items
                if current_row:
                    # Only add padding if it's not the first item in the row
                    current_row_width += item_width + (
                        padding if len(current_row) > 1 else 0
                    )
                else:
                    current_row_width = item_width

        # Add the last row if it's not empty
        if current_row:
            rows.append(current_row)

        # Calculate the height needed for the legend
        row_height = box_size + padding
        required_height = (
            (len(rows) * row_height)
            + ((len(rows) - 1) * vertical_row_padding)
            + padding
        )
        legend_height = max(height, required_height)

        # Create a blank image for the legend with the adjusted height
        legend_image = np.ones((legend_height, width, 3), dtype=np.uint8) * 255

        # Draw each class in the legend
        for row_idx, row in enumerate(rows):
            # Calculate the total width of items in this row
            row_width = sum(item_widths[idx] for idx in row)
            # Add padding between items (number of items - 1)
            row_width += padding * (len(row) - 1)

            # Center the row horizontally
            start_x = (width - row_width) // 2

            # Vertical position for this row
            # (account for vertical padding between rows)
            y_pos = padding + row_idx * (row_height + vertical_row_padding)

            # Reset x position for each row
            x_pos = start_x

            for item_idx in row:
                label, color = classes[item_idx]

                # Process the color based on its type
                if isinstance(color, str) and color.startswith("#"):
                    # Convert hex color string directly to BGR
                    color = hex_to_bgr(color)
                elif isinstance(color, (list, tuple)):
                    # Ensure color is a tuple of exactly 3 integers
                    if len(color) != 3:
                        color = tuple(color[:3])
                    # Convert color values to int if they aren't already
                    color = tuple(int(c) for c in color)
                else:
                    # Default color if something unexpected
                    color = (0, 0, 0)  # Black

                # Draw color box
                cv2.rectangle(
                    legend_image,
                    (x_pos, y_pos),
                    (x_pos + box_size, y_pos + box_size),
                    color,
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
                    (x_pos + box_size + padding, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    line_thickness,
                    cv2.LINE_AA,
                )

                # Move x position for the next item
                # Add the width of the current item + padding for the next item
                x_pos += item_widths[item_idx] + padding

        return legend_image

    @staticmethod
    def _create_header(
        text: str,
        width: int,
        height: int,
        font_scale: float = 2.5,
        padding: int = 10,
        line_thickness: int = 3,
    ) -> np.ndarray:
        """
        Create a header image with text using OpenCV.

        Args:
            text: The header text
            width: Width of the header image
            height: Height of the header image
            font_scale: Scale of the font
            padding: Padding around text
            line_thickness: Thickness of the text lines

        Returns:
            Header image as numpy array in BGR format
        """
        # Create a blank image for the header
        header_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Get text size
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )

        # Calculate center position, accounting for padding
        # Ensure the text remains within the padded area
        text_x = max(padding, (width - text_size[0]) // 2)
        text_y = (height + text_size[1]) // 2

        # Draw text
        cv2.putText(
            header_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            line_thickness,
            cv2.LINE_AA,
        )

        return header_image

    @staticmethod
    def highlight_mask_np(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            mask = np.stack([mask, mask, mask], axis=2)
        img_dimmed = img.copy() * 0.1
        img_lighted = np.clip(img.copy() * 1.2, a_min=0, a_max=255)
        res = img_dimmed * (1 - mask) + (img_lighted - img_dimmed) * mask
        return res

    @staticmethod
    def compose(
        items: list[np.ndarray],
        legend: list[tuple[str, tuple[int, int, int]]] = None,
        header: str = None,
        padding=50,
        header_footer_height=150,
    ) -> np.ndarray:
        assert len(items) > 0
        assert all(2 <= v.ndim <= 3 for v in items)
        assert all(v.shape[:2] == items[0].shape[:2] for v in items)

        items = [
            i if i.ndim == 3 else np.stack([i, i, i], axis=-1) for i in items
        ]
        items = [
            (
                (i * 255).astype(np.uint8)
                if np.max(i) <= 1
                else i.astype(np.uint8)
            )
            for i in items
        ]
        gap = np.full(
            [items[0].shape[0], padding, 3], (255, 255, 255), dtype=np.uint8
        )

        # insert gap between visualizations
        items = [
            item
            for pair in zip(items, [gap] * (len(items) - 1))
            for item in pair
        ] + [items[-1]]

        v = np.pad(
            np.concatenate(items, axis=1),
            ((padding, padding), (padding, padding), (0, 0)),
            constant_values=255,
        )

        if legend is not None:
            # create the legend
            legend_image = SegmVisualizer._create_legend(
                legend,
                width=v.shape[1],
                height=header_footer_height,
            )
            v = np.concatenate((v, legend_image), axis=0)

        if header is not None:
            # create header
            header_image = SegmVisualizer._create_header(
                header,
                width=v.shape[1],
                height=header_footer_height,
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
        header: str,
    ) -> np.ndarray:
        mask_colored = SegmVisualizer.colorize_mask(
            mask,
            classes.colors_map(squeezed=classes_squeezed),
        )

        src = (
            (source * 255).astype(np.uint8)
            if np.max(source) <= 1
            else source.astype(np.uint8)
        )

        overlay = (
            overlay_alpha * (src) + (1 - overlay_alpha) * mask_colored
        ).astype(np.uint8)

        codes = np.unique(mask).tolist()
        if classes_squeezed:
            codes = [classes.idx_to_code[i] for i in codes]

        legend_items = (
            [
                (f"{cl.label} ({cl.name})", cl.color)
                for cl in classes.classes
                if cl.code in codes
            ]
            if show_legend
            else None
        )

        return SegmVisualizer.compose(
            [src, overlay, mask_colored],
            legend=legend_items,
            header=header,
        )

    @staticmethod
    def vis_annotation(
        source: np.ndarray,
        mask: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool = False,
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        return SegmVisualizer._vis_src_with_mask(
            source=source,
            mask=mask,
            classes=classes,
            classes_squeezed=classes_squeezed,
            overlay_alpha=overlay_alpha,
            show_legend=show_legend,
            header="source   |   overlay   |   annotation",
        )

    @staticmethod
    def vis_prediction(
        source: np.ndarray,
        pred: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool = False,
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        return SegmVisualizer._vis_src_with_mask(
            source=source,
            mask=pred,
            classes=classes,
            classes_squeezed=classes_squeezed,
            overlay_alpha=overlay_alpha,
            show_legend=show_legend,
            header="source   |   overlay   |   prediction",
        )

    @staticmethod
    def vis_test(
        source: np.ndarray,
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
            source (np.ndarray): The source image.

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
            values_to_colors={
                0: (244, 67, 54),
                1: (76, 175, 80),
                255: (0, 0, 255),  # void color
            },
        )

        error_overlay = SegmVisualizer.highlight_mask_np(
            source, (mask_gt != mask_pred).astype(np.uint8) * void_mask
        )

        codes_pred = np.unique(mask_pred).tolist()
        codes_gt = np.unique(mask_gt).tolist()

        if mask_pred_squeezed:
            codes_pred = [classes.idx_to_code[i] for i in codes_pred]
            codes_gt = [classes.idx_to_code[i] for i in codes_gt]

        codes = sorted(list(set(codes_gt) | set(codes_pred)))

        legend_items = (
            [
                (f"{cl.label} ({cl.name})", cl.color)
                for cl in classes.classes
                if cl.code in codes
            ]
            if show_legend
            else None
        )

        return SegmVisualizer.compose(
            [source, gt_colored, pred_colored, correct_colored, error_overlay],
            legend=legend_items,
            header=(
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
