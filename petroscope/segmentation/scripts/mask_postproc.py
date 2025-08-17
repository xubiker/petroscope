from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    opening,
    dilation,
)
from skimage.morphology import disk


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
        mask_src = cv2.imread(str(mask_path))[:, :, 0]
        mask_res, mask_diff = mask_morphology(
            mask_src,
            opening_rad=opening_rad,
            extra_dilation_rad=extra_dilation_rad,
            holes_area_threshold=holes_area_threshold,
            small_area_threshold=small_area_threshold,
        )
        mask_src_colored = SegmVisualizer.colorize_mask(
            mask_src,
            classes.colors_map(),
        )
        mask_res_colored = SegmVisualizer.colorize_mask(
            mask_res,
            classes.colors_map(),
        )
        mask_diff_colored = SegmVisualizer.colorize_mask(
            mask_diff,
            classes.colors_map() | {-1: (100, 100, 100)},
        )
        img_compose = SegmVisualizer.compose(
            [mask_src_colored, mask_res_colored, mask_diff_colored],
            header_data="source | result | diff",
        )
        cv2.imwrite(
            str(out_dir / f"{mask_path.stem}_src_colored.png"),
            mask_src_colored,
        )
        cv2.imwrite(
            str(out_dir / f"{mask_path.stem}_res_colored.png"),
            mask_res_colored,
        )
        mask_res = np.stack([mask_res, mask_res, mask_res], axis=-1)
        cv2.imwrite(
            str(out_dir / f"{mask_path.stem}.png"),
            mask_res,
        )
        cv2.imwrite(
            str(out_dir / f"{mask_path.stem}_compare.png"),
            img_compose,
        )


if __name__ == "__main__":

    ds_path = Path("/Users/xubiker/dev/LumenStone/S3_v1.4/")
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True, parents=True)

    classes = LumenStoneClasses.all()
    # samples = ["train", "test", "new"]
    samples = ["new"]

    masks_paths = []
    for sample in samples:
        masks_paths += [
            p
            for p in (ds_path / "masks" / sample).iterdir()
            if p.is_file() and p.suffix == ".png"
        ]

    clean_masks(
        classes,
        masks_paths,
        out_dir,
        opening_rad=1,
        extra_dilation_rad=1,
        holes_area_threshold=18,
    )
