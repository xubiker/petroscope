from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
from petroscope.segmentation.vis import SegmVisualizer


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
    source = np.array(Image.open(img_p))
    mask = np.array(Image.open(mask_p))[:, :, 0]
    vis_img = SegmVisualizer.vis_annotation(
        source=source,
        mask=mask,
        classes=classes,
        classes_squeezed=False,
    )
    vis_img.save(out_p, quality=95)


def vis_mask_colored(
    mask_p: Path,
    out_p: Path,
    classes: ClassSet,
):
    mask = np.array(Image.open(mask_p))[:, :, 0]

    mask_colored = SegmVisualizer.colorize_mask(
        mask,
        classes.colors_map(squeezed=False),
        return_image=True,
    )
    mask_colored.save(out_p)


if __name__ == "__main__":
    datasets_p = {
        # "S1": "/mnt/c/dev/LumenStone/S1_v2/",
        # "S1_v1.0": "/Users/xubiker/dev/LumenStone/S1_v1.0/",
        # "S1_v1.1": "/Users/xubiker/dev/LumenStone/S1_v1.1/",
        # "S1_v1.2": "/Users/xubiker/dev/LumenStone/S1_v1.2/",
        # "S1_v1.3": "/Users/xubiker/dev/LumenStone/S1_v1.3/",
        "S1_v1.5": "/Users/xubiker/dev/LumenStone/S1_v1.5/",
        # "S2": "/Users/xubiker/dev/LumenStone/S2_v1/",
        # "S3": "/Users/xubiker/dev/LumenStone/S3_v1/",
    }

    classes = LumenStoneClasses.S1v1()

    samples = (
        "train",
        "test",
    )

    tasks = []

    for ds in datasets_p.values():
        for sample in samples:
            img_mask_paths = lumenstone_img_mask_paths(Path(ds), sample)
            out_folder_mask = Path(ds) / "masks_colored_png" / sample
            out_folder_human = Path(ds) / "masks_human" / sample
            out_folder_mask.mkdir(exist_ok=True, parents=True)
            out_folder_human.mkdir(exist_ok=True, parents=True)
            for img_p, mask_p in img_mask_paths:
                tasks.append(
                    (
                        img_p,
                        mask_p,
                        out_folder_mask / f"{img_p.stem}.png",
                        out_folder_human / f"{img_p.stem}.jpg",
                    )
                )

    for img_p, mask_p, out_p_mask, out_p_human in tqdm(tasks):
        vis_mask_colored(mask_p, out_p_mask, classes=classes)
        vis_mask_human(img_p, mask_p, out_p_human, classes=classes)
