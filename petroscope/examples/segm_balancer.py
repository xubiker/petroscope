from pathlib import Path
import time

from tqdm import tqdm
from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset

import cv2

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


def run_balancer_sampler(iterations=500, save_patches=True, visualize=True):

    exp_dir = prepare_experiment(Path("./out"))

    ds = ClassBalancedPatchDataset(
        img_mask_paths=img_mask_pairs(
            Path.home() / "dev/LumenStone/S1_v2", "test"
        ),
        patch_size=384,
        class_set=LumenStoneClasses.S1v1(),
        void_border_width=3,
        augment_rotation=20,
        augment_scale=0.1,
        augment_brightness=0.03,
        augment_color=0.02,
        class_area_consideration=1.5,
        patch_positioning_accuracy=0.8,
        balancing_strength=0.75,
        acceleration=8,
        cache_dir=Path.home() / ".petroscope" / "balancer",
    )

    t0 = time.time()
    s = ds.sampler_balanced()
    for i in tqdm(range(iterations), "extracting patches"):
        img, msk = next(s)
        if save_patches:
            (exp_dir / "patches").mkdir(exist_ok=True)
            cv2.imwrite(
                str(exp_dir / f"patches/{i}.jpg"),
                img[:, :, ::-1],
            )
    t1 = time.time()
    print(f"performance sampler: {iterations / (t1 - t0):.2f} patches/s")

    if visualize:
        ds.visualize_probs(out_path=exp_dir / "probs", center_patch=True)
        ds.visualize_accums(out_path=exp_dir / "accums")


def run_balancer_dataloader(
    iterations: int,
    batch_size: int,
    n_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):

    ds = ClassBalancedPatchDataset(
        img_mask_paths=img_mask_pairs(
            Path.home() / "dev/LumenStone/S1_v2", "test"
        ),
        patch_size=384,
        class_set=LumenStoneClasses.S1v1(),
        void_border_width=3,
        augment_rotation=20,
        augment_scale=0.1,
        augment_brightness=0.03,
        augment_color=0.02,
        class_area_consideration=1.5,
        patch_positioning_accuracy=0.8,
        balancing_strength=0.75,
        acceleration=8,
        seed=42,
        cache_dir=Path.home() / ".petroscope" / "balancer",
    )

    dataloader_balanced = ds.dataloader_balanced(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    it = iter(dataloader_balanced)

    t0 = time.time()
    for i in tqdm(range(iterations), "extracting patches"):
        img, msk = next(it)
    t1 = time.time()
    performance = iterations * batch_size / (t1 - t0)
    print(f"performance dataloader: {performance:.2f} patches/s")


if __name__ == "__main__":
    # This fixes the pickling error on macOS and other platforms
    # mp.set_start_method("spawn", force=True)

    n_iterations = 200
    batch_size = 16
    save_patches = False
    visualize = False

    run_balancer_sampler(
        iterations=n_iterations * batch_size,
        save_patches=save_patches,
        visualize=visualize,
    )

    run_balancer_dataloader(
        iterations=n_iterations,
        batch_size=batch_size,
        n_workers=4,
        pin_memory=True,
        prefetch_factor=8,
    )
