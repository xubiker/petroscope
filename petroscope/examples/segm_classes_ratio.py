from pathlib import Path

from PIL import Image

import numpy as np

from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.utils.data import ClassAssociation


def calc_mask_prc(mask_paths: list[Path], classes: ClassAssociation):
    d = dict()
    for mask_p in mask_paths:
        arr = np.array(Image.open(mask_p))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        values, counts = np.unique(arr, return_counts=True)
        for value, count in zip(values, counts):
            if value not in d:
                d[value] = count
            else:
                d[value] += count
    s = float(sum(d.values()))
    d = {classes.code_to_labels[int(v)]: float(c / s) for v, c in d.items()}
    return d


if __name__ == "__main__":
    datasets_p = {
        "S1": "/Users/xubiker/dev/LumenStone/S1_v1/",
        "S2": "/Users/xubiker/dev/LumenStone/S2_v1/",
        "S3": "/Users/xubiker/dev/LumenStone/S3_v1/",
    }

    samples = "train", "test"

    classes = LumenStoneClasses.all()

    for ds_name, ds in datasets_p.items():
        for sample in samples:
            mask_paths = [
                p
                for p in (Path(ds) / "masks" / sample).iterdir()
                if p.is_file() and p.suffix == ".png"
            ]
            classes_ratio = calc_mask_prc(mask_paths, classes)

            print(f"Dataset {ds_name}, {sample}:")
            for k, v in classes_ratio.items():
                print(f"\t{k}: {(v * 100):.3f}%")
