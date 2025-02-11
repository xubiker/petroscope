from pathlib import Path

import hydra

from petroscope.segmentation.balancer import SelfBalancingDataset
from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.models.pspnet.model import PSPNetTorch
from petroscope.segmentation.utils.data import BatchPacker, ClassSet
from petroscope.utils import logger


def test_img_mask_pairs(cfg):
    ds_dir = Path(cfg.data.dataset_path)
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def train_val_samplers(cfg, classes: ClassSet):
    ds_dir = Path(cfg.data.dataset_path)
    train_img_mask_p = [
        (img_p, ds_dir / "masks" / "train" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "train").iterdir())
    ]

    ds_train = SelfBalancingDataset(
        img_mask_paths=train_img_mask_p,
        patch_size=cfg.train.patch_size,
        augment_rotation=cfg.train.augm.rotation,
        augment_scale=cfg.train.augm.scale,
        cls_indices=list(range(16)),
        class_area_consideration=cfg.train.balancer.class_area_consideration,
        patch_positioning_accuracy=cfg.train.balancer.patch_positioning_accuracy,
        balancing_strength=cfg.train.balancer.balancing_strength,
        acceleration=cfg.train.balancer.acceleration,
        cache_dir=Path(cfg.data.cache_path),
        void_border_width=cfg.train.balancer.void_border_width,
    )

    train_sampler_balanced = ds_train.sampler_balanced()
    train_sampler_random = ds_train.sampler_random()

    train_sampler_balanced_batch = iter(
        BatchPacker(
            train_sampler_balanced,
            cfg.train.batch_size,
            classes.code_to_idx,
            normalize_img=True,
            one_hot=False,
        )
    )
    train_sampler_random_batch = iter(
        BatchPacker(
            train_sampler_random,
            cfg.train.batch_size,
            classes.code_to_idx,
            normalize_img=True,
            one_hot=False,
        )
    )

    return (
        train_sampler_balanced_batch,
        train_sampler_random_batch,
        len(ds_train),
    )


@hydra.main(
    version_base="1.2", config_path=".", config_name="train_config.yaml"
)
def run_training(cfg):
    classes = LumenStoneClasses.S1v1()

    train_iterator, val_iterator, ds_len = train_val_samplers(
        cfg, classes=classes
    )

    model = PSPNetTorch(
        n_classes=len(classes),
        backbone=cfg.model.backbone,
        dilated=cfg.model.dilated,
        device=cfg.hardware.device,
    )

    logger.info(model.n_params_str)

    model.train(
        img_mask_paths=None,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        n_steps=ds_len // cfg.train.batch_size * cfg.train.augm.factor,
        LR=cfg.train.LR,
        epochs=cfg.train.epochs,
        val_steps=cfg.train.val_steps,
        test_every=cfg.train.test_every,
        test_params=PSPNetTorch.TestParams(
            classes=classes,
            img_mask_paths=test_img_mask_pairs(cfg),
            void_pad=cfg.test.void_pad,
            void_border_width=cfg.test.void_border_width,
            vis_plots=cfg.test.vis_plots,
            vis_segmentation=cfg.test.vis_segmentation,
        ),
        out_dir=Path("."),
    )


if __name__ == "__main__":
    run_training()
