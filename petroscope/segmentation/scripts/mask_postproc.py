from pathlib import Path
import numpy as np
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    opening,
    dilation,
)
from skimage.morphology import disk


from PIL import Image

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
from petroscope.segmentation.vis import SegmVisualizer

from tqdm import tqdm


def mask_morphology(
    mask: np.ndarray,
    opening_rad: int,
    extra_dilation_rad: int,
    holes_area_threshold: int,
    small_area_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:

    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError("Invalid mask")

    mask_vals = np.unique(mask, return_counts=False)
    mask_vals = set(mask_vals) - set([0])

    filtered_mask = np.zeros_like(mask)

    for v in mask_vals:
        bin_mask = mask == v
        if opening_rad > 0:
            bin_mask = opening(bin_mask, footprint=disk(opening_rad))
        if extra_dilation_rad > 0:
            bin_mask = dilation(bin_mask, footprint=disk(extra_dilation_rad))
        if holes_area_threshold > 0:
            bin_mask = remove_small_holes(bin_mask, holes_area_threshold)
        if small_area_threshold > 0:
            bin_mask = remove_small_objects(bin_mask, small_area_threshold)
        filtered_mask = np.where(bin_mask, v, filtered_mask)
    diff = np.where(filtered_mask != mask, filtered_mask, -1)

    return filtered_mask, diff


def clean_masks(
    classes: ClassSet,
    masks_paths: list[Path],
    out_dir: Path,
    opening_rad: int = 3,
    extra_dilation_rad: int = 3,
    holes_area_threshold: int = 25,
    small_area_threshold: int = 20,
):
    for mask_path in tqdm(masks_paths):
        mask_src = np.array(Image.open(mask_path), dtype=np.uint8)[:, :, 0]
        mask_res, mask_diff = mask_morphology(
            mask_src,
            opening_rad=opening_rad,
            extra_dilation_rad=extra_dilation_rad,
            holes_area_threshold=holes_area_threshold,
            small_area_threshold=small_area_threshold,
        )
        mask_src_colored = SegmVisualizer.colorize_mask(
            mask_src,
            classes.colors_map(squeezed=False),
            return_image=False,
        )
        mask_res_colored = SegmVisualizer.colorize_mask(
            mask_res,
            classes.colors_map(squeezed=False),
            return_image=False,
        )
        mask_diff_colored = SegmVisualizer.colorize_mask(
            mask_diff,
            classes.colors_map(squeezed=False) | {-1: (100, 100, 100)},
            return_image=False,
        )
        img_compose = SegmVisualizer.compose(
            [mask_src_colored, mask_res_colored, mask_diff_colored],
            header="source | result | diff",
        )
        Image.fromarray(mask_src_colored).save(
            out_dir / f"{mask_path.stem}_src_colored.png"
        )

        Image.fromarray(mask_res_colored).save(
            out_dir / f"{mask_path.stem}_res_colored.png"
        )
        mask_res = np.stack([mask_res, mask_res, mask_res], axis=-1)
        Image.fromarray(mask_res).save(out_dir / f"{mask_path.stem}.png")

        img_compose.save(out_dir / f"{mask_path.stem}_compare.png")


if __name__ == "__main__":

    ds_path = Path("/Users/xubiker/dev/LumenStone/S1_v2_old/")
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True, parents=True)

    classes = LumenStoneClasses.S1v1()
    masks_paths = [
        p
        for p in (ds_path / "masks" / "train").iterdir()
        if p.is_file() and p.suffix == ".png"
    ] + [
        p
        for p in (ds_path / "masks" / "test").iterdir()
        if p.is_file() and p.suffix == ".png"
    ]

    new_names = [
        "train_23",
        "train_60",
        "train_61",
        "train_62",
        "train_63",
        "train_64",
        "test_18",
        "test_19",
        "test_20",
    ]

    masks_paths_new = [p for p in masks_paths if p.stem in new_names]
    masks_paths_old = [p for p in masks_paths if p.stem not in new_names]

    clean_masks(
        classes,
        masks_paths_new,
        out_dir,
        opening_rad=3,
        extra_dilation_rad=3,
        holes_area_threshold=18,
    )
    clean_masks(
        classes,
        masks_paths_old,
        out_dir,
        opening_rad=0,
        extra_dilation_rad=0,
        holes_area_threshold=18,
    )
