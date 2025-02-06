from pathlib import Path
from PIL import Image
from typing import Iterable

import numpy as np
from tqdm import tqdm

from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.utils.data import ClassSet
from petroscope.segmentation.utils.vis import SegmVisualizer


def lumenstone_img_mask_paths(
    ds_folder: Path, sample="train"
) -> list[tuple[Path, Path]]:
    return [
        (f, ds_folder / "masks" / sample / f"{f.stem}.png")
        for f in (ds_folder / "imgs" / sample).iterdir()
        if f.is_file() and f.suffix == ".jpg"
    ]


def vis_mask_human(
    img_p: Path,
    mask_p: Path,
    out_p: Path,
    classes: ClassSet,
):
    vis_img = SegmVisualizer.vis_annotation(
        source=np.array(Image.open(img_p)),
        mask=np.array(Image.open(mask_p))[:, :, 0],
        classes=classes,
        classes_squeezed=False,
    )
    vis_img.save(out_p, quality=95)


if __name__ == "__main__":
    datasets_p = {
        "S1": "/mnt/c/dev/LumenStone/S1_v1_x05/",
        # "S1": "/Users/xubiker/dev/LumenStone/S1_v1/",
        # "S2": "/Users/xubiker/dev/LumenStone/S2_v1/",
        # "S3": "/Users/xubiker/dev/LumenStone/S3_v1/",
    }

    classes = LumenStoneClasses.S1v1()

    samples = "train", "test"

    tasks = []

    for ds in datasets_p.values():
        for sample in samples:
            img_mask_paths = lumenstone_img_mask_paths(Path(ds), sample)
            out_dir = Path(ds) / "masks_human" / sample
            out_dir.mkdir(exist_ok=True, parents=True)
            for img_p, mask_p in img_mask_paths:
                tasks.append((img_p, mask_p, out_dir / f"{img_p.stem}.jpg"))

    for img_p, mask_p, out_p in tqdm(tasks):
        vis_mask_human(img_p, mask_p, out_p, classes=classes)
