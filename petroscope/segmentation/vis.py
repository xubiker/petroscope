from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go

from petroscope.segmentation.classes import ClassSet
from petroscope.segmentation.loggers import TrainingLogger
from petroscope.utils import logger


def _rgb_to_hex(color_rgb: tuple[float, float, float]) -> str:
    r = int(color_rgb[0] * 255)
    g = int(color_rgb[1] * 255)
    b = int(color_rgb[2] * 255)
    color_hex = f"rgb({r},{g},{b})"
    return color_hex


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
            classes.colors_map(),
        )

        overlay = cv2.addWeighted(
            src1=source,
            alpha=overlay_alpha,
            src2=mask_colored,
            beta=1 - overlay_alpha,
            gamma=0,
        )

        codes = np.unique(mask).tolist()

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
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Visualize a segmentation annotation on top of the source image.

        Args:
            source_bgr: The source image (BGR format)
            mask: The segmentation mask (2D array of class codes)
            classes: The set of segmentation classes
            overlay_alpha: The alpha value for the overlay (between 0 and 1)
            show_legend: If True, show a legend with class names and colors

        Returns:
            The visualization as a numpy array in BGR format
        """
        return SegmVisualizer._vis_src_with_mask(
            source=source_bgr,
            mask=mask,
            classes=classes,
            overlay_alpha=overlay_alpha,
            show_legend=show_legend,
            header_data="source   |   overlay   |   annotation",
        )

    @staticmethod
    def vis_prediction(
        source_bgr: np.ndarray,
        pred: np.ndarray,
        classes: ClassSet,
        overlay_alpha: float = 0.8,
        show_legend: bool = True,
    ) -> np.ndarray:
        return SegmVisualizer._vis_src_with_mask(
            source=source_bgr,
            mask=pred,
            classes=classes,
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

            show_legend (bool, optional): Whether to show the legend in the
            visualization. Defaults to True.

        Returns:
            np.ndarray: The composite visualization image.
        """
        pred_colored = SegmVisualizer.colorize_mask(
            mask_pred,
            classes.colors_map(),
        )

        gt_colored = SegmVisualizer.colorize_mask(
            mask_gt,
            classes.colors_map(),
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
    def plot_values_over_epochs(
        data: list[tuple[int, float]] | dict[str, list[tuple[int, float]]],
        out_dir: Path,
        title: str,
        y_axis_label: str,
        filename: str,
        x_axis_label: str = "Epoch",
        line_color: str = None,
        plot_size: tuple[int, int] = (3000, 1500),
        colors: dict[str, tuple[float, float, float]] = None,
        legend_position: dict = None,
    ):
        """
        Generic function to plot values over epochs and save as an image.
        Can plot single or multiple curves on the same plot.

        Args:
            data: Either a list of (epoch, value) tuples for a single curve,
                or a dict mapping curve names to lists of (epoch, value) tuples
                for multiple curves.
            out_dir (Path): The path where the image will be saved.
            title (str): The title of the plot.
            y_axis_label (str): The label for the y-axis.
            filename (str): The filename for the saved image (no extension).
            x_axis_label (str, optional): The label for the x-axis.
                Defaults to "Epoch".
            line_color (str, optional): Color for the line. Only used for
                single curve plots. If None, uses default.
            plot_size (tuple[int, int], optional): The size of the plot,
                defaults to (3000, 1500).
            colors (dict[str, tuple[float, float, float]], optional):
                Custom color mapping for multi-class plots. Keys should match
                curve names.
            legend_position (dict, optional): Custom legend positioning.
        """
        fig = go.Figure()

        # Handle single curve (list) or multiple curves (dict)
        if isinstance(data, dict):
            # Multiple curves
            default_colors = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
            ]
            for i, (curve_name, curve_data) in enumerate(data.items()):
                if colors and curve_name in colors:
                    # Use custom RGB color
                    color = _rgb_to_hex(colors[curve_name])
                else:
                    # Use default color
                    color = default_colors[i % len(default_colors)]

                # Extract x and y values from (epoch, value) tuples
                x_values = [epoch for epoch, value in curve_data]
                y_values = [value for epoch, value in curve_data]

                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name=curve_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                    )
                )
        else:
            # Single curve - data is list of (epoch, value) tuples
            x_values = [epoch for epoch, value in data]
            y_values = [value for epoch, value in data]

            trace_kwargs = {
                "x": x_values,
                "y": y_values,
                "mode": "lines+markers",
                "line": dict(width=4),
                "marker": dict(size=6),
            }
            if line_color:
                trace_kwargs["line"]["color"] = line_color
                trace_kwargs["marker"]["color"] = line_color

            fig.add_trace(go.Scatter(**trace_kwargs))

        layout_args = {
            "title": title,
            "xaxis_title": x_axis_label,
            "yaxis_title": y_axis_label,
            "width": plot_size[0],
            "height": plot_size[1],
            "font": dict(size=18),
            "template": "plotly_white",
        }

        fig.update_layout(**layout_args)

        # Add legend if multiple curves
        if isinstance(data, dict):
            if legend_position:
                fig.update_layout(legend=legend_position)
            else:
                fig.update_layout(
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=16),
                    )
                )

        fig.write_image(out_dir / f"{filename}.png")

    @staticmethod
    def plot_losses(tl: TrainingLogger, out_dir: Path) -> None:
        """
        Generate training and validation loss plots using data
        from JSON logger.
        Args:
            tl (TrainingLogger): TrainingLogger instance containing data
            out_dir (Path): Directory to save plots
        """
        try:
            # Get data from JSON logger
            train_losses_data = tl.get_losses("train_loss")
            val_losses_data = tl.get_losses("val_loss")

            # Convert to (epoch, value) tuples
            train_data = [
                (epoch, loss) for epoch, loss in train_losses_data.items()
            ]
            val_data = [
                (epoch, loss) for epoch, loss in val_losses_data.items()
            ]

            # Plot combined losses
            Plotter.plot_values_over_epochs(
                data={
                    "Training Loss": train_data,
                    "Validation Loss": val_data,
                },
                out_dir=out_dir,
                title="Training and Validation Loss",
                y_axis_label="Loss",
                filename="loss",
            )

            logger.info("Combined loss plot saved")

        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")

    @staticmethod
    def plot_lrs(tl: TrainingLogger, out_dir: Path) -> None:
        """
        Generate learning rate schedule plot using data from JSON logger.
        Args:
            tl (TrainingLogger): TrainingLogger instance containing data
            out_dir (Path): Directory to save plots
        """
        try:
            # Get data from JSON logger
            learning_rates_data = tl.get_lr()

            # Convert to (epoch, value) tuples
            lr_data = [
                (epoch, lr) for epoch, lr in learning_rates_data.items()
            ]
            Plotter.plot_values_over_epochs(
                data=lr_data,
                out_dir=out_dir,
                title="Learning Rate Schedule",
                y_axis_label="Learning Rate",
                filename="lrs",
            )
            logger.info("Learning rate plot saved")

        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")

    @staticmethod
    def plot_metrics(
        tl: TrainingLogger,
        void: bool,
        out_dir: Path,
        colors: dict[str, tuple[float, float, float]],
    ) -> None:
        """Plot segmentation metrics over epochs.

        Args:
            tl (TrainingLogger): JSON logger instance containing metrics data
            void (bool): Whether to include void class in metrics
            out_dir (Path): Directory to save plots
        """
        metrics_data = tl.get_metrics(void=void)
        suffix = "_void" if void else "_full"
        Plotter._plot_combined_metrics(
            metrics_data,
            out_dir,
            suffix=suffix,
        )
        Plotter._plot_class_metrics(
            metrics_data,
            out_dir,
            colors=colors,
            suffix=suffix,
        )

    @staticmethod
    def _plot_combined_metrics(
        metrics_data: dict[int, dict], out_dir: Path, suffix: str = ""
    ) -> None:
        """
        Plot a set of metrics (either with or without void).

        Args:
            metrics_data: Dictionary of epoch -> metrics data
            out_dir: Directory to save plots
            suffix: Suffix to add to filenames
        """
        if not metrics_data:
            return

        # Extract PA, mIoU hard, and mIoU soft data
        pa_data = []
        miou_hard_data = []
        miou_soft_data = []

        for epoch, data in metrics_data.items():
            # PA data
            if "PA" in data:
                pa_data.append((epoch, data["PA"]))

            # Hard metrics
            if "hard" in data:
                hard_metrics = data["hard"]
                if "mIoU" in hard_metrics:
                    miou_hard_data.append((epoch, hard_metrics["mIoU"]))

            # Soft metrics
            if "soft" in data:
                soft_metrics = data["soft"]
                if "mIoU" in soft_metrics:
                    miou_soft_data.append((epoch, soft_metrics["mIoU"]))

        # Plot combined PA, mIoU hard, mIoU soft
        if pa_data and miou_hard_data and miou_soft_data:
            Plotter.plot_values_over_epochs(
                data={
                    "Pixel Accuracy": pa_data,
                    "mIoU (Hard)": miou_hard_data,
                    "mIoU (Soft)": miou_soft_data,
                },
                out_dir=out_dir,
                title=f"Training Metrics{suffix.replace('_', ' ').title()}",
                y_axis_label="Score",
                filename=f"metrics{suffix}",
            )
            logger.info(f"Combined metrics plot{suffix} saved")

    @staticmethod
    def _plot_class_metrics(
        metrics_data: dict[int, dict],
        out_dir: Path,
        suffix: str = "",
        colors: dict[str, tuple[float, float, float]] = None,
    ) -> None:
        """
        Plot a set of metrics (either with or without void).

        Args:
            metrics_data: Dictionary of epoch -> metrics data
            out_dir: Directory to save plots
            suffix: Suffix to add to filenames
        """
        if not metrics_data:
            return

        # Extract per-class IoU data for hard and soft metrics
        hard_per_class = {}
        soft_per_class = {}

        for epoch, data in metrics_data.items():
            # Hard metrics
            if "hard" in data:
                hard_metrics = data["hard"]

                # Per-class hard IoU
                if "IoU_per_class" in hard_metrics:
                    for class_name, iou_value in hard_metrics[
                        "IoU_per_class"
                    ].items():
                        if class_name not in hard_per_class:
                            hard_per_class[class_name] = []
                        hard_per_class[class_name].append((epoch, iou_value))

            # Soft metrics
            if "soft" in data:
                soft_metrics = data["soft"]

                # Per-class soft IoU
                if "IoU_per_class" in soft_metrics:
                    for class_name, iou_value in soft_metrics[
                        "IoU_per_class"
                    ].items():
                        if class_name not in soft_per_class:
                            soft_per_class[class_name] = []
                        soft_per_class[class_name].append((epoch, iou_value))

        # Plot per-class hard IoU
        if hard_per_class:
            title_suffix = suffix.replace("_", " ").title()
            Plotter.plot_values_over_epochs(
                data=hard_per_class,
                out_dir=out_dir,
                title=f"Per-Class IoU (Hard){title_suffix}",
                y_axis_label="IoU",
                colors=colors,
                filename=f"iou_hard{suffix}",
            )
            logger.info(f"Hard IoU plot{suffix} saved")

        # Plot per-class soft IoU
        if soft_per_class:
            title_suffix = suffix.replace("_", " ").title()
            Plotter.plot_values_over_epochs(
                data=soft_per_class,
                out_dir=out_dir,
                title=f"Per-Class IoU (Soft){title_suffix}",
                y_axis_label="IoU",
                colors=colors,
                filename=f"iou_soft{suffix}",
            )
            logger.info(f"Soft IoU plot{suffix} saved")
