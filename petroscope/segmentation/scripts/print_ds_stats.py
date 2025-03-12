from pathlib import Path

import numpy as np
from PIL import Image

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
        arr = np.array(Image.open(mask_p))
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


def print_dataset_stats(
    ds_name: str,
    ds_path: Path,
    classes: ClassSet,
    samples: tuple[str],
):
    d = {}
    n_pixels_total = 0
    for sample in samples:
        mask_paths = [
            p
            for p in (ds_path / "masks" / sample).iterdir()
            if p.is_file() and p.suffix == ".png"
        ]
        n_pixels, class_dict = calc_mask_ratio(mask_paths, classes)
        n_pixels_total += n_pixels
        for cls, n in class_dict.items():
            if cls in d:
                d[cls][sample] = n
            else:
                d[cls] = {sample: n}

    s = ", ".join(([s for s in samples] + ["total"]))
    print(f"Dataset {ds_name} [{s}]:")
    for cls, v in d.items():
        prc = [(v.get(sample, 0) / n_pixels_total) * 100 for sample in samples]
        prc.append(sum(prc))
        q = ", ".join([f"{p:.2f}%" for p in prc])
        print(f"\t\t {cls}: {q}")


if __name__ == "__main__":
    datasets_p = {
        "S1": "/Users/xubiker/dev/LumenStone/S1_v1.5/",
        "S2": "/Users/xubiker/dev/LumenStone/S2_v1/",
        # "S3": "/mnt/c/dev/LumenStone/S3_v1/",
    }
    samples = ("train", "test")

    classes = LumenStoneClasses.all()

    for ds_name, ds_path in datasets_p.items():
        print_dataset_stats(ds_name, Path(ds_path), classes, samples)
