from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from petroscope.segmentation.metrics import SegmMetrics


def hex_to_rgb(hex: str):
    """
    Converts a hexadecimal color code to its RGB representation.

    Args:
        hex (str): The hexadecimal color code to convert.

    Returns:
        tuple: A tuple of three integers representing the RGB values of the color.
    """
    h = hex.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def to_heat_map(img: np.ndarray, name="jet"):
    """
    Converts a 2D image to a heat map.

    Args:
        img (numpy.ndarray): The input image to convert. It must be a 2D array.
        name (str, optional): The name of the color map to use. Defaults to "jet".

    Returns:
        numpy.ndarray: The heat map image as a 3D array with shape (height, width, 3).
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
    def _load_as_array(a, dtype=np.uint8) -> np.ndarray | None:
        match a:
            case np.ndarray():
                return a.astype(dtype)
            case Image.Image():
                return np.array(a).astype(dtype)
            case Path():
                return np.array(Image.open(a)).astype(dtype)
            case _:
                return None

    @staticmethod
    def colorize_mask(
        mask: np.ndarray,
        idx_to_colors: dict[int, tuple[int, int, int]],
        return_image=False,
    ) -> np.ndarray | Image.Image:
        """
        This function colorizes a segmentation mask based on the provided class indices to colors mapping.

        Args:
            mask (np.ndarray): The input segmentation mask to colorize.
            idx_to_colors (dict[int, tuple[int, int, int]]): A dictionary mapping class indices to their corresponding RGB colors.
            return_image (bool, optional): Whether to return the colorized mask as a PIL Image. Defaults to False.

        Returns:
            np.ndarray | Image.Image: The colorized segmentation mask as a 3D numpy array or a PIL Image if return_image is True.
        """
        colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for code, color in idx_to_colors.items():
            colorized[mask == code, :] = color
        if return_image:
            return Image.fromarray(colorized)
        return colorized

    @staticmethod
    def overlay(
        mask: np.ndarray,
        overlay: np.ndarray | Image.Image | Path = None,
        alpha=0.75,
    ) -> Image.Image:
        """
        Overlay a mask on an image or another mask.

        Args:
            mask (np.ndarray): The mask to be overlaid. It should have 3 channels.
            overlay (np.ndarray | Image.Image | Path, optional): The image or mask to be overlaid on the mask. Defaults to None.
            alpha (float, optional): The transparency of the overlay. Defaults to 0.75.

        Returns:
            Image.Image: The resulting image with the mask overlaid on the overlay.

        """

        assert mask.ndim == 3, "only 3-channel masks are supported"

        if overlay is not None:
            overlay = SegmVisualizer._load_as_array(overlay)
            assert overlay.shape[:2] == mask.shape[:2]
            assert overlay.ndim == 3

        overlay_res = Image.fromarray(
            (alpha * overlay + (1 - alpha) * mask).astype(np.uint8)
        )
        return overlay_res

    @staticmethod
    def to_image(mask: np.ndarray) -> Image.Image:
        """
        Convert a numpy array mask to a PIL Image object.

        Args:
            mask (np.ndarray): The input mask to be converted. It should be a 2D or 3D numpy array of dtype np.uint8.

        Returns:
            Image.Image: The converted PIL Image object.

        """
        assert mask.ndim in (2, 3) and mask.dtype == np.uint8
        return Image.fromarray((mask).astype(np.uint8))


class Plotter:
    """
    This class contains static methods for plotting various metrics and learning rate schedules.
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
            name_suffix (str, optional): A suffix to append to the metric name. Defaults to "".

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
            metrics (Iterable[SegmMetrics]): An iterable of SegmMetrics objects containing the metrics to be plotted.
            out_dir (Path): The output directory where the plots will be saved.
            colors (dict[str, tuple[float, float, float]]): A dictionary mapping class labels to their RGB colors.
            name_suffix (str, optional): A suffix to be added to the plot filenames. Defaults to "".

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
