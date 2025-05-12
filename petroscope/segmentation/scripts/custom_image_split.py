from pathlib import Path

import cv2
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
        img = cv2.imread(str(p))
        if img is None:
            print(f"Failed to read image {p}. Skipping...")
            continue

        h, w, _ = img.shape
        if w < n_cols * patch_size or h < n_rows * patch_size:
            print(f"Image {p.name} is too small. Skipping...")
            continue

        y0 = (h - n_rows * patch_size) // 2
        x0 = (w - n_cols * patch_size) // 2

        for idx in range(n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            y1 = y0 + i * patch_size
            x1 = x0 + j * patch_size
            patch = img[y1 : y1 + patch_size, x1 : x1 + patch_size]

            patch_path = folder_out / f"{p.stem}_{i}_{j}.jpg"
            cv2.imwrite(
                str(patch_path), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )


if __name__ == "__main__":

    folder_in = Path("/mnt/c/dev/LumenStone/ICM1_v1/imgs")
    folder_out = Path("/mnt/c/dev/LumenStone/ICM1_v1/imgs_cropped")

    img_paths = [
        f for f in folder_in.iterdir() if f.is_file() and f.suffix == ".jpg"
    ]

    split(img_paths, folder_out)
