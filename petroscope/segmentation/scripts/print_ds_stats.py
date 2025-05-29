from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses


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

    d = dict()
    total_pixels = 0
    for mask_p in mask_paths:
        arr = cv2.imread(str(mask_p))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        total_pixels += arr.shape[0] * arr.shape[1]
        values, counts = np.unique(arr, return_counts=True)
        for value, count in zip(values, counts):
            if value not in d:
                d[value] = count
            else:
                d[value] += count
    d = {classes.code_to_label[int(v)]: c for v, c in d.items()}
    return (total_pixels, d)


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


def calc_stats(datasets: list[Dataset]):
    # calculate statistics for each dataset and sample
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
            n_pixels, class_to_count_dict = calc_mask_ratio(
                mask_paths, classes
            )
            statistics[dataset.name][sample] = DatasetSampleStatistics(
                dataset, sample, n_pixels, class_to_count_dict
            )
    return statistics


def print_stats(
    statistics: dict[str, dict[str, DatasetSampleStatistics]],
    sum_pixels_across_datasets=True,
    round_to: int = 2,
):
    # normalize accordning to sum_pixels_across_datasets
    # and print stats
    n_total = None
    if sum_pixels_across_datasets:
        n_total = sum(
            [
                sum(
                    [
                        ds_sample_stats.total_pixels
                        for ds_sample_stats in d.values()
                    ]
                )
                for d in statistics.values()
            ]
        )

    global_sum = 0
    global_sum_round = 0

    for ds_name in statistics:
        samples = list(statistics[ds_name].keys())
        print("Dataset ", ds_name, f"[{', '.join(['total'] + samples)}]:")

        # calc total number of pixels in all samples of the dataset
        # if sum_pixels_across_datasets is False
        if not sum_pixels_across_datasets:
            n_total = sum(
                [
                    ds_sample_stats.total_pixels
                    for ds_sample_stats in statistics[ds_name].values()
                ]
            )

        prc = dict()  # class -> sample -> prc
        for sample, stat in statistics[ds_name].items():
            for cls, n in stat.class_to_count_dict.items():
                if cls not in prc:
                    prc[cls] = dict()
                prc[cls][sample] = (n / n_total) * 100

        for cls, sample_to_prc_dict in prc.items():
            q = [
                float(sample_to_prc_dict.get(sample, 0)) for sample in samples
            ]
            total_prc = round(sum(q), round_to)
            global_sum += sum(q)
            global_sum_round += total_prc
            qq = [str(round(i, round_to)) + "%" for i in q]
            print(f"\t\t {cls}: {total_prc}% [{', '.join(qq)}]")

    print(f"Total: {global_sum}, round: {global_sum_round}")


if __name__ == "__main__":
    datasets_p = {
        "S1": Path.home() / "dev/LumenStone/S1_v2/",
        "S2": Path.home() / "dev/LumenStone/S2_v2/",
        "S3": Path.home() / "dev/LumenStone/S3_v1/",
    }
    # samples = ("all",)
    samples = ("train", "test")

    classes = LumenStoneClasses.all()

    datasets = [
        Dataset(ds_name, ds_path, classes=classes, samples=samples)
        for ds_name, ds_path in datasets_p.items()
    ]

    stats = calc_stats(datasets)
    print_stats(stats, sum_pixels_across_datasets=True, round_to=1)

    # for ds_name, ds_path in datasets_p.items():
    #     print_dataset_stats(ds_name, Path(ds_path), classes, samples)
