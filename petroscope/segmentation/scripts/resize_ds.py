from pathlib import Path

import cv2
from tqdm import tqdm


def resize(
    img_path: Path,
    mask_path: Path,
    img_dir_out: Path,
    mask_dir_out: Path,
    factor: float = 0.5,
) -> None:

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    h, w = img.shape[:2]
    mask_h, mask_w = mask.shape[:2]

    assert w == mask_w and h == mask_h

    new_w = int(w * factor)
    new_h = int(h * factor)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(img_dir_out / img_path.name), img)
    cv2.imwrite(str(mask_dir_out / mask_path.name), mask)


def simple_resize(
    img_path: Path,
    img_dir_out: Path,
    factor: float = 0.5,
) -> None:

    img = cv2.imread(str(img_path))

    h, w = img.shape[:2]

    new_w = int(w * factor)
    new_h = int(h * factor)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(img_dir_out / img_path.name), img)


def resize_segm_ds(
    ds_dir: Path,
    ds_dir_out: Path,
    samples: tuple[str] = ("train", "test"),
    factor: float = 0.5,
) -> None:
    """
    Resize all images and masks in the dataset directory by a given factor.
    """

    for sample in samples:

        img_dir = ds_dir / "imgs" / sample
        mask_dir = ds_dir / "masks" / sample

        img_mask_p = [
            (img_p, mask_dir / f"{img_p.stem}.png")
            for img_p in sorted(img_dir.iterdir())
        ]

        img_dir_out = ds_dir_out / "imgs" / sample
        mask_dir_out = ds_dir_out / "masks" / sample

        img_dir_out.mkdir(parents=True, exist_ok=True)
        mask_dir_out.mkdir(parents=True, exist_ok=True)

        for img_p, mask_p in tqdm(img_mask_p, f"resizing {sample}"):
            resize(img_p, mask_p, img_dir_out, mask_dir_out, factor=factor)


def resize_panorama_ds(
    ds_dir: Path, ds_dir_out: Path, factor: float = 0.5
) -> None:
    """
    Resize all images in the panorama dataset directory by a given factor.
    """
    img_p = [p for p in sorted(ds_dir.iterdir())]

    ds_dir_out.mkdir(parents=True, exist_ok=True)

    for p in tqdm(img_p, desc="Resizing images"):
        simple_resize(p, ds_dir_out, factor=factor)


if __name__ == "__main__":

    # resize dataset with segmentation masks
    # resize_segm_ds(
    #     ds_dir=Path.home() / "dev/LumenStone/S1v2_S2v2",
    #     ds_dir_out=Path.home() / "dev/LumenStone/S1v2_S2v2_x075",
    #     factor=0.75,
    # )

    # resize panorams
    resize_panorama_ds(
        ds_dir=Path.home() / "dev/LumenStone/P1_results",
        ds_dir_out=Path.home() / "dev/LumenStone/P1_results_x05",
        factor=0.5,
    )
