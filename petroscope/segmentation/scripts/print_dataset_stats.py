"""
Script for analyzing and printing statistics of existing segmentation datasets.

This script analyzes mask images and provides detailed statistics about class
distributions across different datasets and samples.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses


def calc_single_mask_ratio(
    mask_path: Path,
    classes: ClassSet,
) -> dict[str, int]:
    """
    Calculate the count of each class label in a single mask image.

    Args:
        mask_path (Path): Path to the mask image file.
        classes (ClassSet): An object that provides a mapping from class
        codes to labels.

    Returns:
        dict[str, int]: A dictionary mapping class labels to their
        pixel counts.
    """
    arr = cv2.imread(str(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    values, counts = np.unique(arr, return_counts=True)
    d = {}
    for value, count in zip(values, counts):
        if int(value) in classes.code_to_label:
            label = classes.code_to_label[int(value)]
            d[label] = count
    return d


def calc_mask_ratio(
    mask_paths: list[Path],
    classes: ClassSet,
) -> tuple[int, dict[str, int]]:
    """
    Calculate the total number of pixels and the count of each class label
    in a list of mask images.

    Args:
        mask_paths (list[Path]): A list of paths to mask image files.
        classes (ClassSet): An object that provides a mapping from class
        codes to labels.

    Returns:
        tuple[int, dict[str, int]]: A tuple where the first element is the
        total number of pixels across all mask images, and the second element
        is a dictionary mapping class labels to their respective pixel counts.
    """
    total_pixels = 0
    total_class_counts = {}

    for mask_path in mask_paths:
        # Get class counts for this mask
        class_counts = calc_single_mask_ratio(mask_path, classes)

        # Add pixel count (need to load image to get dimensions)
        arr = cv2.imread(str(mask_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        total_pixels += arr.shape[0] * arr.shape[1]

        # Accumulate class counts
        for class_name, count in class_counts.items():
            total_class_counts[class_name] = (
                total_class_counts.get(class_name, 0) + count
            )

    return total_pixels, total_class_counts


@dataclass
class Dataset:
    name: str
    path: Path
    classes: ClassSet
    samples: tuple[str]


@dataclass
class DatasetSampleStatistics:
    dataset: Dataset
    sample: str
    total_pixels: int
    class_to_count_dict: dict


def calc_stats(
    datasets: list[Dataset],
) -> dict[str, dict[str, DatasetSampleStatistics]]:
    """
    Calculate statistics for each dataset and sample.

    Args:
        datasets: List of Dataset objects to analyze

    Returns:
        dict: Nested dictionary with structure:
        {dataset_name: {sample_name: DatasetSampleStatistics}}
    """
    statistics = dict()
    for dataset in datasets:
        statistics[dataset.name] = {}
        for sample in dataset.samples:
            print(f"Scanning dataset {dataset.name}, sample {sample}...")
            mask_paths = [
                p
                for p in (dataset.path / "masks" / sample).iterdir()
                if p.is_file() and p.suffix == ".png"
            ]
            if not mask_paths:
                print(f"  No masks found in {dataset.path / 'masks' / sample}")
                continue

            n_pixels, class_to_count_dict = calc_mask_ratio(
                mask_paths, dataset.classes
            )
            statistics[dataset.name][sample] = DatasetSampleStatistics(
                dataset, sample, n_pixels, class_to_count_dict
            )
            print(f"  Found {len(mask_paths)} mask files")
    return statistics


def print_stats(
    statistics: dict[str, dict[str, DatasetSampleStatistics]],
    sum_pixels_across_datasets: bool = True,
    round_to: int = 2,
) -> None:
    """
    Print formatted statistics for datasets.

    Args:
        statistics: Statistics dictionary from calc_stats()
        sum_pixels_across_datasets: If True, calculate percentages across
        all datasets. If False, calculate per-dataset percentages.
        round_to: Number of decimal places for rounding percentages
    """
    if not statistics:
        print("No statistics to display")
        return

    # Calculate total pixels for normalization
    n_total = None
    if sum_pixels_across_datasets:
        n_total = sum(
            [
                sum(
                    [
                        ds_sample_stats.total_pixels
                        for ds_sample_stats in d.values()
                        if ds_sample_stats is not None
                    ]
                )
                for d in statistics.values()
            ]
        )

    global_sum = 0
    global_sum_round = 0

    for ds_name in statistics:
        samples = list(statistics[ds_name].keys())
        if not samples:
            continue

        print("Dataset ", ds_name, f"[{', '.join(['total'] + samples)}]:")

        # calc total number of pixels in all samples of the dataset
        # if sum_pixels_across_datasets is False
        if not sum_pixels_across_datasets:
            n_total = sum(
                [
                    ds_sample_stats.total_pixels
                    for ds_sample_stats in statistics[ds_name].values()
                    if ds_sample_stats is not None
                ]
            )

        if n_total == 0:
            print("  No data available")
            continue

        prc = dict()  # class -> sample -> prc
        for sample, stat in statistics[ds_name].items():
            if stat is None:
                continue
            for cls, n in stat.class_to_count_dict.items():
                if cls not in prc:
                    prc[cls] = dict()
                prc[cls][sample] = (n / n_total) * 100

        # Sort classes for consistent display
        for cls in sorted(prc.keys()):
            sample_to_prc_dict = prc[cls]
            q = [
                float(sample_to_prc_dict.get(sample, 0)) for sample in samples
            ]
            total_prc = round(sum(q), round_to)
            global_sum += sum(q)
            global_sum_round += total_prc
            qq = [str(round(i, round_to)) + "%" for i in q]
            print(f"\t\t {cls}: {total_prc}% [{', '.join(qq)}]")

    print(f"Total: {global_sum:.{round_to}f}, rounded: {global_sum_round}")


def analyze_predefined_datasets() -> None:
    """Analyze predefined datasets with hardcoded paths."""
    datasets_p = {
        # "S1": Path.home() / "dev/LumenStone/S1_v2/",
        # "S2": Path.home() / "dev/LumenStone/S2_v2/",
        "S3": Path.home()
        / "dev/LumenStone/S3_v1.3/",
    }
    samples = ("train", "test")
    classes = LumenStoneClasses.all()

    datasets = [
        Dataset(ds_name, ds_path, classes=classes, samples=samples)
        for ds_name, ds_path in datasets_p.items()
        if ds_path.exists()
    ]

    if not datasets:
        print("No predefined datasets found")
        return

    stats = calc_stats(datasets)
    print_stats(stats)


if __name__ == "__main__":
    analyze_predefined_datasets()
