from pathlib import Path

from tqdm import tqdm
from petroscope.segmentation.balancer import SelfBalancingDataset

from PIL import Image


# provide path to LumenStone dataset here
ds_dir = Path("/mnt/c/dev/LumenStone/S1_v1")

img_dir = ds_dir / "imgs" / "train"
mask_dir = ds_dir / "masks" / "train"
img_mask_p = [
    (img_p, mask_dir / f"{img_p.stem}.png")
    for img_p in sorted(img_dir.iterdir())
]

ds = SelfBalancingDataset(
    img_mask_paths=img_mask_p,
    patch_size=256,
    augment_rotation=30,
    # augment_scale=0.1,
    cls_indices=list(range(16)),
    class_area_consideration=1.5,
    patch_positioning_accuracy=0.8,
    balancing_strength=0.75,
    acceleration=8,
    cache_dir=Path(".") / "cache",
)

save_patches = True

iterations = 1000
s = ds.sampler_balanced()
for i in tqdm(range(iterations), "extracting patches"):

    img, msk = next(s)
    if save_patches:
        Path("./out/patches/").mkdir(exist_ok=True)
        Image.fromarray(img).save(f"./out/patches/{i}.jpg")


print(ds.accum)
ds.visualize_probs(out_path=Path("./out/probs/"), center_patch=True)
ds.visualize_accums(out_path=Path("./out/accums/"))
