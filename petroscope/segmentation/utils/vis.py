from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from petroscope.segmentation.metrics import SegmMetrics


def hex_to_rgb(hex: str):
    h = hex.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def to_heat_map(img, name="jet"):
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
        codes_to_colors: dict[int, tuple[int, int, int]],
        return_image=False,
    ) -> np.ndarray | Image.Image:
        colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for code, color in codes_to_colors.items():
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
        assert mask.ndim in (2, 3) and mask.dtype == np.uint8
        return Image.fromarray((mask).astype(np.uint8))


class Plotter:

    @staticmethod
    def plot_single_class_metric(
        out_dir: Path,
        metric_name: str,
        values: Iterable[float],
    ):
        epochs = len(values)
        fig = plt.figure(figsize=(12, 6))
        # ax = plt.axes()
        # ax.set_facecolor('white')
        x = [x + 1 for x in range(epochs)]
        y = [values[i] for i in range(epochs)]
        plt.plot(x, y)
        # plt.suptitle(f'{metric_name} over epochs', fontsize=20)
        plt.ylabel(f"{metric_name}", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        fig.savefig(out_dir / f"{metric_name}.png")

    @staticmethod
    def plot_multi_class_metric(
        out_dir: Path,
        metric_name,
        data: dict[str, Iterable[float]],
        colors: dict[str, tuple[float, float, float]],
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
        plt.ylabel(f"{metric_name}", fontsize=20)
        plt.xlabel("epoch", fontsize=20)
        plt.legend(
            [cl_str for cl_str in data], loc="center right", fontsize=15
        )
        fig.savefig(out_dir / f"{metric_name}_per_class.png")

    @staticmethod
    def plot_segm_metrics(
        metrics: Iterable[SegmMetrics],
        out_dir: Path,
        colors: dict[str, tuple[float, float, float]],
    ):

        labels = metrics[0].iou.keys()

        # transform metrics data to plot data
        single_class_plot_data = {
            "acc": [m.acc.value for m in metrics],
            "mean_iou": [m.mean_iou_soft for m in metrics],
            "mean_iou_strict": [m.mean_iou for m in metrics],
        }
        multi_class_plot_data = {
            "iou": {
                label: [m.iou_soft[label].value for m in metrics]
                for label in labels
            },
            "iou_strict": {
                label: [m.iou[label].value for m in metrics]
                for label in labels
            },
        }

        # perform plotting
        for metric_name, data in single_class_plot_data.items():
            Plotter.plot_single_class_metric(out_dir, metric_name, data)
        for metric_name, data in multi_class_plot_data.items():
            Plotter.plot_multi_class_metric(
                out_dir,
                metric_name,
                data,
                colors=colors,
            )

    @staticmethod
    def plot_lrs(lrs: list, output_path: Path):
        plt.style.use("ggplot")
        fig = plt.figure()
        plt.plot([i + 1 for i in range(0, len(lrs))], lrs)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        fig.savefig(output_path / "lrs.jpg")
        plt.close()
