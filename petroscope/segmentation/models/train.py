from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig

from petroscope.segmentation.balancer import ClassBalancedPatchDataset
from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.segmentation.models.resunet.model import ResUNet
from petroscope.segmentation.models.pspnet.model import PSPNet
from petroscope.segmentation.models.upernet.model import UPerNet
from petroscope.segmentation.utils import BasicBatchCollector
from petroscope.utils import logger


def set_pytorch_seed(seed: int):
    """Fix all pytorch seeds for reproducibility."""
    import os

    # Set environment variable for PyTorch to enable deterministic algorithms
    # (should be set before importing torch)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from petroscope.utils.lazy_imports import torch  # noqa

    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.__version__ >= "1.8.0":
        # Use deterministic algorithms where available (PyTorch 1.8+)
        # This is a warning-only setting and will not raise an error
        # if deterministic algorithms are not available
        torch.use_deterministic_algorithms(True, warn_only=True)


def test_img_mask_pairs(cfg: DictConfig):
    """Get test image-mask pairs from the dataset."""
    ds_dir = Path(cfg.data.dataset_path)
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def train_val_samplers(
    cfg: DictConfig,
    use_dataloaders: bool = True,
):
    """Create training and validation data samplers."""
    ds_dir = Path(cfg.data.dataset_path)
    train_img_mask_p = [
        (img_p, ds_dir / "masks" / "train" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "train").iterdir())
    ]

    ds_train = ClassBalancedPatchDataset(
        img_mask_paths=train_img_mask_p,
        patch_size=cfg.train.patch_size,
        augment_rotation=cfg.train.augm.rotation,
        augment_scale=cfg.train.augm.scale,
        augment_brightness=cfg.train.augm.brightness,
        augment_color=cfg.train.augm.color,
        class_set=LumenStoneClasses.from_name(cfg.data.classes),
        class_area_consideration=cfg.train.balancer.class_area_consideration,
        patch_positioning_accuracy=cfg.train.balancer.patch_positioning_accuracy,
        balancing_strength=cfg.train.balancer.balancing_strength,
        acceleration=cfg.train.balancer.acceleration,
        cache_dir=Path(cfg.data.cache_path),
        void_border_width=cfg.train.balancer.void_border_width,
        seed=cfg.hardware.seed,
    )

    balanced = cfg.train.balancer.enabled

    logger.warning(f"Using {'balanced' if balanced else 'random'} sampling")

    train_it = None
    val_it = None

    if use_dataloaders:
        train_sampler = (
            ds_train.dataloader_balanced(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=cfg.train.data_loader.pin_memory,
            )
            if balanced
            else ds_train.dataloader_random(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=cfg.train.data_loader.pin_memory,
            )
        )
        val_sampler = ds_train.dataloader_random(
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.data_loader.num_workers,
            prefetch_factor=cfg.train.data_loader.prefetch_factor,
            pin_memory=cfg.train.data_loader.pin_memory,
        )

        train_it = iter(train_sampler)
        val_it = iter(val_sampler)
    else:
        train_sampler = (
            ds_train.sampler_balanced()
            if balanced
            else ds_train.sampler_random()
        )
        val_sampler = ds_train.sampler_random()

        train_it = iter(
            BasicBatchCollector(
                train_sampler,
                cfg.train.batch_size,
            )
        )
        val_it = iter(
            BasicBatchCollector(
                val_sampler,
                cfg.train.batch_size,
            )
        )

    return (train_it, val_it, len(ds_train))


def create_model(
    model_type: Literal["resunet", "pspnet", "upernet"],
    cfg: DictConfig,
    n_classes: int,
) -> PatchSegmentationModel:
    """
    Create a segmentation model based on the specified type.

    Args:
        model_type: Type of model to create ("resunet", "pspnet", or "upernet")
        cfg: Configuration object
        n_classes: Number of segmentation classes

    Returns:
        Initialized model
    """
    if model_type == "resunet":
        return ResUNet(
            n_classes=n_classes,
            device=cfg.hardware.device,
            filters=cfg.model.resunet.filters,
            layers=cfg.model.resunet.layers,
        )
    elif model_type == "pspnet":
        return PSPNet(
            n_classes=n_classes,
            backbone=cfg.model.pspnet.backbone,
            dilated=cfg.model.pspnet.dilated,
            device=cfg.hardware.device,
        )
    elif model_type == "upernet":
        return UPerNet(
            n_classes=n_classes,
            backbone=cfg.model.upernet.backbone,
            device=cfg.hardware.device,
            use_fpn=cfg.model.upernet.get("use_fpn", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def run_training(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """

    # Set the random seed for reproducibility
    set_pytorch_seed(cfg.hardware.seed)

    # Get model type from config or use default
    model_type = cfg.get("model_type", "resunet")

    # Load class definitions
    classes = LumenStoneClasses.from_name(cfg.data.classes)

    # Create data samplers
    train_iterator, val_iterator, ds_len = train_val_samplers(cfg)

    # Create model
    model = create_model(model_type, cfg, len(classes))

    logger.info(f"Training {model_type.upper()} model")
    logger.info(model.n_params_str)

    # Train the model
    model.train(
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        n_steps=ds_len // cfg.train.batch_size * cfg.train.augm.factor,
        LR=cfg.train.LR,
        scheduler_patience=cfg.train.scheduler_patience,
        epochs=cfg.train.epochs,
        val_steps=cfg.train.val_steps,
        test_every=cfg.train.test_every,
        test_params=model.TestParams(
            classes=classes,
            img_mask_paths=test_img_mask_pairs(cfg),
            void_pad=cfg.test.void_pad,
            void_border_width=cfg.test.void_border_width,
            vis_segmentation=cfg.test.vis_segmentation,
            max_epoch_visualizations=cfg.test.max_epoch_visualizations,
        ),
        out_dir=Path("."),
        amp=cfg.get("amp", False),
        gradient_clipping=cfg.get("gradient_clipping", 1.0),
    )


if __name__ == "__main__":
    run_training()
