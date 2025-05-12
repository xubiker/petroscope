from pathlib import Path

from tqdm import tqdm
from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset

from PIL import Image

from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.utils.base import prepare_experiment


def img_mask_pairs(ds_dir: Path, sample: str) -> list[tuple[Path, Path]]:
    img_dir = ds_dir / "imgs" / sample
    mask_dir = ds_dir / "masks" / sample
    img_mask_p = [
        (img_p, mask_dir / f"{img_p.stem}.png")
        for img_p in sorted(img_dir.iterdir())
    ]
    return img_mask_p


def run_balancer(iterations=1000, save_patches=True):

    exp_dir = prepare_experiment(Path("./out"))

    ds = ClassBalancedPatchDataset(
        img_mask_paths=img_mask_pairs(
            Path.home() / "dev/LumenStone/S1_v2", "test"
        ),
        patch_size=256,
        class_set=LumenStoneClasses.S1v1(),
        mask_classes_mapping={0: 10, 1: 20, 8: 35},
        void_border_width=3,
        augment_rotation=None,
        augment_scale=None,
        class_area_consideration=1.5,
        patch_positioning_accuracy=0.8,
        balancing_strength=0.75,
        acceleration=8,
        cache_dir=Path.home() / ".petroscope" / "balancer",
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
