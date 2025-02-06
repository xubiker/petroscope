from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from petroscope.segmentation.metrics import SegmMetrics
from petroscope.segmentation.utils.data import ClassSet


def hex_to_rgb(hex: str):
    """
    Converts a hexadecimal color code to its RGB representation.

    Args:
        hex (str): The hexadecimal color code to convert.

    Returns:
        tuple: A tuple of three integers representing
        the RGB values of the color.
    """
    h = hex.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def to_heat_map(img: np.ndarray, name="jet"):
    """
    Converts a 2D image to a heat map.

    Args:
        img (numpy.ndarray): The input image to convert. It must be a 2D array.
        name (str, optional): The name of the color map to use.Defaults
        to "jet".

    Returns:
        numpy.ndarray: The heat map image as a 3D array
        with shape (height, width, 3).
    """
    assert img.ndim == 2, "shape {} is unsupported".format(img.shape)
    img_min, img_max = np.min(img), np.max(img)
    assert (
        img_min >= 0.0 and img_max <= 1.0
    ), f"invalid range {img_min} - {img_max}"
    img = img / img_max if img_max != 0 else img
    cmap = plt.get_cmap(name)
    heat_img = cmap(img)[..., 0:3]
    return (heat_img * 255).astype(np.uint8)


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
            return np.array(Image.open(a), dtype=dtype)
        else:
            raise TypeError(f"Unsupported type for loading: {type(a)}")

    @staticmethod
    def colorize_mask(
        mask: np.ndarray,
        values_to_colors: dict[int, tuple[int, int, int]],
        return_image=False,
    ) -> np.ndarray | Image.Image:
        """
        This function colorizes a segmentation mask based on the provided
        class indices to colors mapping.

        Args:
            mask (np.ndarray): The input segmentation mask to colorize.

            values_to_colors (dict[int, tuple[int, int, int]]): A dictionary
            mapping mask values to corresponding RGB colors.

            return_image (bool, optional): Whether to return the colorized
            mask as a PIL Image. Defaults to False.

        Returns:
            np.ndarray | Image.Image: The colorized segmentation mask as a 3D
            numpy array or a PIL Image if return_image is True.
        """
        colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for code, color in values_to_colors.items():
            colorized[mask == code, :] = color
        if return_image:
            return Image.fromarray(colorized)
        return colorized

    @staticmethod
    def overlay(
        mask: np.ndarray,
        overlay: np.ndarray | Image.Image | Path = None,
        alpha=0.6,
    ) -> Image.Image:
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
            Image.Image: The resulting image with the mask overlaid on
            the overlay.

        """

        assert mask.ndim == 3, "only 3-channel masks are supported"

        if overlay is not None:
            overlay = SegmVisualizer._load_as_array(overlay)
            assert overlay.shape[:2] == mask.shape[:2]
            assert overlay.ndim == 3
        else:
            overlay = mask.copy()

        overlay_res = Image.fromarray(
            np.clip(
                (alpha * overlay + (1 - alpha) * mask).astype(np.uint8),
                0,
                255,
            )
        )
        return overlay_res

    @staticmethod
    def _create_legend(
        classes,
        width: int,
        height: int,
        font_size: int = 100,
        box_size: int = 100,
        padding: int = 25,
    ):
        """
        Create a legend as a separate image.

        :param classes: List of (label, color) tuples.
        :param width: Width of the legend image.
        :param height: Height of the legend area.
        :param font_size: Initial font size for the legend text.
        :param box_size: Size of the color boxes.
        :param padding: Padding between elements.
        :return: Pillow Image object containing the legend.
        """
        # Create a blank image for the legend
        legend_image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(legend_image)

        # Load font
        font = ImageFont.load_default(font_size)

        # Resize legend elements if needed
        while True:
            # Calculate total legend width
            total_width = 0
            for label, color in classes:
                text_bbox = font.getbbox(label)
                text_width = text_bbox[2] - text_bbox[0]
                total_width += (
                    box_size + padding + text_width + padding * 2
                )  # Box + text + paddings

            total_width += 2 * padding

            # Check if it fits within the width
            if total_width <= width:
                break

            # Reduce sizes if the legend is too wide
            font_size -= 5
            box_size -= 5
            if font_size < 6 or box_size < 6:
                raise ValueError(
                    "Legend cannot fit within the specified width."
                )
            font = ImageFont.load_default(font_size)  # Reload the font

        # Draw the legend centered horizontally
        start_x = (width - total_width) // 2
        start_y = (height - box_size) // 2

        for label, color in classes:
            # Draw the color box
            draw.rectangle(
                [start_x, start_y, start_x + box_size, start_y + box_size],
                fill=color,
            )
            # Draw the label
            text_bbox = font.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = start_x + box_size + padding
            draw.text((text_x, start_y), label, fill=(0, 0, 0), font=font)
            # Update x-coordinate for the next class
            start_x = text_x + text_width + padding * 2

        return legend_image

    @staticmethod
    def _create_header(
        text: str,
        width: int,
        height: int,
        font_size: int = 100,
        padding: int = 25,
    ):
        """
        Create a legend as a separate image.

        :param text: Header text.
        :param width: Width of the header area.
        :param height: Height of the header area.
        :param font_size: Initial font size for the header text.
        :param padding: Padding between elements.
        :return: Pillow Image object containing the header.
        """
        # create a blank image for the header
        header_image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(header_image)

        # load font
        font = ImageFont.load_default(font_size)

        # resize header elements if needed
        while True:
            # calculate total header width
            total_width = 0
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            total_width = text_width + padding * 2
            total_height = text_height + padding * 2

            # check if it fits within the width and height
            if (total_width <= width) and (total_height <= height):
                break

            # reduce sizes if the header is too wide
            font_size -= 5
            if font_size < 6:
                raise ValueError(
                    "Header cannot fit within the specified width."
                )
            font = ImageFont.load_default(font_size)  # Reload the font

        # draw the header centered
        start_x = (width - text_width) // 2
        start_y = (height - text_height) // 2
        draw.text((start_x, start_y), text, fill=(0, 0, 0), font=font)

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
    ) -> Image.Image:
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
            v = np.concatenate((v, np.array(legend_image)), axis=0)

        if header is not None:
            # create header
            header_image = SegmVisualizer._create_header(
                header,
                width=v.shape[1],
                height=header_footer_height,
            )
            v = np.concatenate((np.array(header_image), v), axis=0)

        return Image.fromarray(v)

    @staticmethod
    def _vis_src_with_mask(
        source: np.ndarray,
        mask: np.ndarray,
        classes: ClassSet,
        classes_squeezed: bool,
        overlay_alpha: float,
        show_legend: bool,
        header: str,
    ) -> Image.Image:
        mask_colored = SegmVisualizer.colorize_mask(
            mask,
            classes.colors_map(squeezed=classes_squeezed),
            return_image=False,
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
    ) -> Image.Image:
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
    ) -> Image.Image:
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
    ):
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
            return_image=False,
        )

        gt_colored = SegmVisualizer.colorize_mask(
            mask_gt,
            classes.colors_map(squeezed=mask_gt_squeezed),
            return_image=False,
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
            return_image=False,
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
