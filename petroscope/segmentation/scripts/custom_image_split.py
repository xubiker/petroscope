from pathlib import Path

from PIL import Image
from tqdm import tqdm


def split(
    img_paths: list[Path],
    folder_out: Path,
    patch_size=1132,
    n_rows=2,
    n_cols=3,
) -> None:
    folder_out.mkdir(parents=True, exist_ok=True)

    for p in tqdm(img_paths):
        img = Image.open(p)
        # print check size
        w, h = img.size
        if w < n_cols * patch_size or h < n_rows * patch_size:
            print(f"Image {p.name} is too small. Skipping...")

        y0 = (h - n_rows * patch_size) // 2
        x0 = (w - n_cols * patch_size) // 2

        for idx in range(n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            patch = img.crop(
                (
                    x0 + j * patch_size,
                    y0 + i * patch_size,
                    x0 + (j + 1) * patch_size,
                    y0 + (i + 1) * patch_size,
                )
            )
            patch.save(folder_out / f"{p.stem}_{i}_{j}.jpg", quality=95)


if __name__ == "__main__":

    folder_in = Path("/mnt/c/dev/LumenStone/ICM1_v1/imgs")
    folder_out = Path("/mnt/c/dev/LumenStone/ICM1_v1/imgs_cropped")

    img_paths = [
        f for f in folder_in.iterdir() if f.is_file() and f.suffix == ".jpg"
    ]

    split(img_paths, folder_out)
