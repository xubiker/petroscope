from pathlib import Path

import cv2
from tqdm import tqdm

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
from petroscope.segmentation.utils import load_image, load_mask
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
    source = load_image(img_p)
    mask = load_mask(mask_p)
    vis_img = SegmVisualizer.vis_annotation(
        source_bgr=source[:, :, ::-1],
        mask=mask,
        classes=classes,
    )
    cv2.imwrite(str(out_p), vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def vis_mask_colored(
    mask_p: Path,
    out_p: Path,
    classes: ClassSet,
):
    mask = load_mask(mask_p)

    mask_colored = SegmVisualizer.colorize_mask(
        mask,
        classes.colors_map(),
    )
    cv2.imwrite(
        str(out_p),
        mask_colored,
    )


if __name__ == "__main__":
    datasets_p = {
        # "S1v1": Path.home() / "dev/LumenStone/S1_v1/",
        # "S1v2": Path.home() / "dev/LumenStone/S1_v2/",
        # "S2v1": Path.home() / "dev/LumenStone/S2_v1/",
        "S3v1": Path.home()
        / "dev/LumenStone/S3_v1.5/",
    }

    classes = LumenStoneClasses.all()

    samples = (
        "train",
        "test",
    )

    tasks = []

    for ds in datasets_p.values():
        for sample in samples:
            img_mask_paths = lumenstone_img_mask_paths(Path(ds), sample)
            out_folder_mask = Path(ds) / "masks_colored" / sample
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
