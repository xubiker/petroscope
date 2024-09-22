from pathlib import Path

from tqdm import tqdm
from petroscope.segmentation.balancer.balancer import SelfBalancingDataset

from PIL import Image

from petroscope.segmentation.utils.base import prepare_experiment


def img_mask_pairs(ds_dir: Path):
    img_dir = ds_dir / "imgs" / "train"
    mask_dir = ds_dir / "masks" / "train"
    img_mask_p = [
        (img_p, mask_dir / f"{img_p.stem}.png")
        for img_p in sorted(img_dir.iterdir())
    ]
    return img_mask_p


def run_balancer(iterations=1000, save_patches=True):

    exp_dir = prepare_experiment(Path("./out"))

    ds = SelfBalancingDataset(
        img_mask_paths=img_mask_pairs(Path("/mnt/c/dev/LumenStone/S1_v1")),
        patch_size=256,
        augment_rotation=30,
        augment_scale=0.1,
        cls_indices=list(range(16)),
        class_area_consideration=1.5,
        patch_positioning_accuracy=0.8,
        balancing_strength=0.75,
        acceleration=8,
        cache_dir=Path(".") / "cache",
    )

    s = ds.sampler_balanced()
    for i in tqdm(range(iterations), "extracting patches"):

        img, msk = next(s)
        if save_patches:
            (exp_dir / "patches").mkdir(exist_ok=True)
            Image.fromarray(img).save(exp_dir / f"patches/{i}.jpg")

    print(ds.accum)
    ds.visualize_probs(out_path=exp_dir / "probs", center_patch=True)
    ds.visualize_accums(out_path=exp_dir / "accums")


if __name__ == "__main__":
    run_balancer()
