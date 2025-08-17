from pathlib import Path
from typing import Literal
import warnings

import hydra
from omegaconf import DictConfig

from petroscope.segmentation.balancer import ClassBalancedPatchDataset
from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.segmentation.models.resunet.model import ResUNet
from petroscope.segmentation.models.pspnet.model import PSPNet
from petroscope.segmentation.models.hrnet import HRNet
from petroscope.segmentation.utils import (
    BasicBatchCollector,
    get_img_mask_pairs,
)
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
        # Suppress warning for operations without deterministic implementation
        warnings.filterwarnings(
            "ignore",
            message=".*nll_loss2d_forward_out_cuda_template does not have a deterministic implementation.*",
        )
        torch.use_deterministic_algorithms(True, warn_only=True)


def create_train_dataset(cfg: DictConfig, class_set=None):
    """Create a training dataset.

    Args:
        cfg: Configuration object
        class_set: Optional ClassSet to use instead of loading from config

    Returns:
        ClassBalancedPatchDataset instance
    """
    ds_dir = Path(cfg.data.dataset_path)
    train_img_mask_p = get_img_mask_pairs(ds_dir, "train")

    # Use provided class_set or load from config
    if class_set is None:
        class_set = LumenStoneClasses.from_name(cfg.data.classes)

    return ClassBalancedPatchDataset(
        img_mask_paths=train_img_mask_p,
        patch_size=cfg.train.patch_size,
        augment_rotation=cfg.train.augm.rotation,
        augment_scale=cfg.train.augm.scale,
        augment_brightness=cfg.train.augm.brightness,
        augment_color=cfg.train.augm.color,
        class_set=class_set,
        class_area_consideration=cfg.train.balancer.class_area_consideration,
        patch_positioning_accuracy=(
            cfg.train.balancer.patch_positioning_accuracy
        ),
        balancing_strength=cfg.train.balancer.balancing_strength,
        acceleration=cfg.train.balancer.acceleration,
        cache_dir=Path(cfg.data.cache_path),
        void_border_width=cfg.train.balancer.void_border_width,
        seed=cfg.hardware.seed,
    )


def create_samplers(
    dataset: ClassBalancedPatchDataset,
    cfg: DictConfig,
    use_dataloaders: bool = True,
):
    """Create training and validation data samplers from a dataset.

    Args:
        dataset: The dataset to create samplers from
        cfg: Configuration object
        use_dataloaders: Whether to use PyTorch DataLoaders

    Returns:
        Tuple of (train_iterator, val_iterator, dataset_length)
    """
    balanced = cfg.train.balancer.enabled
    logger.info(f"Using {'balanced' if balanced else 'random'} sampling")

    train_it = None
    val_it = None

    if use_dataloaders:
        device = cfg.hardware.device
        use_pin_memory = cfg.train.data_loader.pin_memory

        # Setup pin_memory_device when using CUDA
        pin_memory_device = None
        if use_pin_memory and "cuda:" in device:
            # Extract device index for better compatibility
            device_idx = int(device.split(":")[-1])
            pin_memory_device = f"cuda:{device_idx}"
            logger.info(
                f"Using pin_memory with device: {device} (index: {device_idx})"
            )

        # Create dataloaders with proper device config
        train_sampler = (
            dataset.dataloader_balanced(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=use_pin_memory,
                pin_memory_device=pin_memory_device,
            )
            if balanced
            else dataset.dataloader_random(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=use_pin_memory,
                pin_memory_device=pin_memory_device,
            )
        )

        val_sampler = dataset.dataloader_random(
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.data_loader.num_workers,
            prefetch_factor=cfg.train.data_loader.prefetch_factor,
            pin_memory=use_pin_memory,
            pin_memory_device=pin_memory_device,
        )

        train_it = iter(train_sampler)
        val_it = iter(val_sampler)
    else:
        train_sampler = (
            dataset.sampler_balanced()
            if balanced
            else dataset.sampler_random()
        )
        val_sampler = dataset.sampler_random()

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

    return (train_it, val_it, len(dataset))


def create_model(
    model_type: Literal["resunet", "pspnet", "hrnet"],
    cfg: DictConfig,
    n_classes: int,
) -> PatchSegmentationModel:
    """
    Create a segmentation model based on the specified type.

    Args:
        model_type: Type of model to create ("resunet", "pspnet" or "hrnet")
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
            backbone=cfg.model.resunet.get("backbone", None),
            dilated=cfg.model.resunet.get("dilated", False),
            pretrained=cfg.model.resunet.get("pretrained", True),
        )
    elif model_type == "pspnet":
        return PSPNet(
            n_classes=n_classes,
            backbone=cfg.model.pspnet.backbone,
            dilated=cfg.model.pspnet.dilated,
            device=cfg.hardware.device,
        )
    elif model_type == "hrnet":
        return HRNet(
            n_classes=n_classes,
            device=cfg.hardware.device,
            backbone=cfg.model.hrnet.backbone,
            pretrained=cfg.model.hrnet.pretrained,
            ocr_mid_channels=cfg.model.hrnet.ocr_mid_channels,
            dropout=cfg.model.hrnet.dropout,
            use_aux_head=cfg.model.hrnet.use_aux_head,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_manager(cfg: DictConfig, dataset, device: str):
    """
    Create a loss manager based on the configuration.

    Args:
        cfg: Configuration object
        dataset: Dataset with get_class_pixel_counts method
        device: Device for tensor operations

    Returns:
        Initialized LossManager

    Raises:
        ValueError: If loss configuration is missing or invalid
    """
    from petroscope.segmentation.losses import LossManager

    # Get loss configuration
    loss_config = cfg.train.get("loss")
    if loss_config is None:
        raise ValueError("Loss configuration is required in training config")

    # Get class pixel counts from dataset
    class_counts = dataset.get_class_pixel_counts()
    logger.info(f"Using class counts for loss weighting: {class_counts}")

    return LossManager.from_config_and_dataset(loss_config, dataset, device)


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def run_training(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """

    # Set the random seed for reproducibility
    set_pytorch_seed(cfg.hardware.seed)

    # Get model type from config
    model_type = cfg["model_type"]

    # Load class definitions with auto-detection support
    if cfg.data.classes == "auto":
        logger.info("Auto-detecting classes from training dataset...")

        # Get training image-mask pairs for scanning
        ds_dir = Path(cfg.data.dataset_path)
        train_img_mask_paths = get_img_mask_pairs(ds_dir, "train")

        # Auto-detect the classes (only known ones)
        classes = LumenStoneClasses.auto_from_dataset(train_img_mask_paths)

        logger.info(f"Auto-detected classes:\n{classes}")

        # Create the training dataset with detected classes
        train_ds = create_train_dataset(cfg, class_set=classes)
    else:
        classes = LumenStoneClasses.from_name(cfg.data.classes)
        logger.info(f"Using predefined class set: {cfg.data.classes}")
        logger.info(f"{classes}")

        # Create the training dataset
        train_ds = create_train_dataset(cfg)

    # Create data samplers using the dataset
    train_iterator, val_iterator, ds_len = create_samplers(train_ds, cfg)

    # Create model - use max_classes from LumenStone class definitions
    model = create_model(model_type, cfg, LumenStoneClasses.max_classes())

    logger.info(f"Training {model_type.upper()} model")
    logger.info(model.n_params_str)

    # Get loss configuration
    loss_config = cfg.train.get("loss", None)

    # Get class counts for loss weight calculation
    class_counts = (
        train_ds.get_class_pixel_counts()
        if hasattr(train_ds, "get_class_pixel_counts")
        else None
    )

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
            img_mask_paths=get_img_mask_pairs(
                Path(cfg.data.dataset_path), "test"
            ),
            void_pad=cfg.test.void_pad,
            void_border_width=cfg.test.void_border_width,
            vis_segmentation=cfg.test.vis_segmentation,
            max_epoch_visualizations=cfg.test.max_epoch_visualizations,
        ),
        out_dir=Path("."),
        amp=cfg.get("amp", False),
        gradient_clipping=cfg.get("gradient_clipping", 1.0),
        loss_config=loss_config,
        class_counts=class_counts,  # Pass class counts instead of the dataset
    )


if __name__ == "__main__":
    run_training()
