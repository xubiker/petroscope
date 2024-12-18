from pathlib import Path

from PIL import Image

import numpy as np
from tqdm import tqdm

from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.utils.vis import SegmVisualizer


def calc_mask_prc(mask_paths: list[Path]):
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
    d = {int(v): float(c / s) for v, c in d.items()}
    return d


def get_img_mask_paths(
    ds_folder: Path, sample="train"
) -> list[tuple[Path, Path]]:
    return [
        (f, ds_folder / "masks" / sample / f"{f.stem}.png")
        for f in (ds_folder / "imgs" / sample).iterdir()
        if f.is_file() and f.suffix == ".jpg"
    ]


def visualize_mask_composite(img_p: Path, mask_p: Path, out_p: Path):
    img = np.array(Image.open(img_p))
    mask = np.array(Image.open(mask_p))[:, :, 0]
    visualization = SegmVisualizer.composite_visualization(
        mask=mask,
        assoc=LumenStoneClasses.all(),
        image=img,
    )
    visualization.save(out_p, quality=95)


if __name__ == "__main__":

    datasets_p = {
        # "S1": "/Users/xubiker/dev/LumenStone/S1_v1/",
        # "S2": "/Users/xubiker/dev/LumenStone/S2_v1/",
        "S3": "/Users/xubiker/dev/LumenStone/S3_v1/",
    }

    samples = "train", "test"

    tasks = []

    for ds in datasets_p.values():
        for sample in samples:
            img_mask_paths = get_img_mask_paths(Path(ds), sample)
            out_dir = Path(ds) / "masks_human" / sample
            out_dir.mkdir(exist_ok=True, parents=True)
            for img_p, mask_p in img_mask_paths:
                tasks.append((img_p, mask_p, out_dir / f"{img_p.stem}.jpg"))

    for img_p, mask_p, out_p in tqdm(tasks):
        visualize_mask_composite(img_p, mask_p, out_p)
