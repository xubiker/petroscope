from pathlib import Path

from PIL import Image
from tqdm import tqdm


def resize(
    img_path: Path,
    mask_path: Path,
    img_dir_out: Path,
    mask_dir_out: Path,
    factor: float = 0.5,
) -> None:

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    w, h = img.size

    assert w == mask.size[0] and h == mask.size[1]

    w = int(w * factor)
    h = int(h * factor)

    img = img.resize((w, h), resample=Image.BILINEAR)
    mask = mask.resize((w, h), resample=Image.NEAREST)

    img.save(img_dir_out / img_path.name)
    mask.save(mask_dir_out / mask_path.name)


if __name__ == "__main__":

    ds_dir = Path("/mnt/c/dev/LumenStone/S2_v1_calib")
    ds_dir_out = Path("/mnt/c/dev/LumenStone/S2_v1_x05_calib")

    samples = "train", "test"

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
            resize(img_p, mask_p, img_dir_out, mask_dir_out, factor=0.5)
