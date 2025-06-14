from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import requests
from tqdm import tqdm

from petroscope.segmentation.classes import ClassSet
from petroscope.segmentation.eval import SegmDetailedTester
from petroscope.segmentation.loggers import DetailedTestLogger, TrainingLogger
from petroscope.segmentation.models.abstract import GeoSegmModel
from petroscope.segmentation.vis import Plotter

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

from petroscope.utils import logger
from petroscope.utils.lazy_imports import nn, optim, torch  # noqa


class PatchSegmentationModel(GeoSegmModel):
    """
    Base class for patch-based segmentation models with common training
    pipeline.

    This class implements common functionality shared between different
    segmentation models like ResUNet, PSPNet, and UPerNet that are trained
    on image patches. It provides methods for training, prediction, and
    model evaluation.

    Subclasses should implement:
    - MODEL_REGISTRY: A dictionary mapping model names to weight URLs
    - __init__: Initialize the specific model architecture
    - _create_from_checkpoint: Class method to create a model instance from
    checkpoint data
    - _get_checkpoint_data: Return model-specific data for checkpoint saving
    """

    MODEL_REGISTRY: dict[str, str] = {}  # Maps model names to weight URLs
    CACHE_DIR = Path.home() / ".petroscope" / "models"

    @dataclass
    class TestParams:
        classes: ClassSet
        img_mask_paths: Iterable[tuple[str, str]]
        void_pad: int
        void_border_width: int
        vis_segmentation: bool
        max_epoch_visualizations: int  # Keep last N epoch visualizations

    def __init__(self, n_classes: int, device: str) -> None:
        """
        Initialize the patch segmentation model.

        Args:
            n_classes: Number of segmentation classes
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.model = None  # To be set by subclasses
        self.tester: SegmDetailedTester | None = None

    @classmethod
    def from_pretrained(
        cls, source: str, device: str, force_download: bool = False
    ) -> "PatchSegmentationModel":
        """
        Load a trained model from either a registry name or local
        checkpoint path.

        Args:
            source: Either a model name from MODEL_REGISTRY or path to
                local checkpoint
            device: Device to load the model on
            force_download: Whether to force download (only for registry
                models)

        Returns:
            Initialized and loaded model
        """
        # Check if it's a registry model name
        if source in cls.MODEL_REGISTRY:
            return cls._from_registry(source, device, force_download)

        # Otherwise treat as local path
        checkpoint_path = Path(source)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model '{source}' not found in registry and path doesn't "
                f"exist. Available registry models: "
                f"{list(cls.MODEL_REGISTRY.keys())}"
            )

        return cls._from_local_checkpoint(checkpoint_path, device)

    @classmethod
    def _from_registry(
        cls, model_name: str, device: str, force_download: bool = False
    ) -> "PatchSegmentationModel":
        """Load model from registry (private method)."""
        weights_url = cls.MODEL_REGISTRY[model_name]
        weights_path = (
            Path.home() / ".cache" / "petroscope" / f"{model_name}.pth"
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # Download if not available
        if not weights_path.exists() or force_download:
            logger.info(f"Downloading weights for {model_name}...")
            cls._download_weights(weights_url, weights_path)

        return cls._from_local_checkpoint(weights_path, device)

    @classmethod
    def _from_local_checkpoint(
        cls, checkpoint_path: Path, device: str
    ) -> "PatchSegmentationModel":
        """Load model from local checkpoint (private method)."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls._create_from_checkpoint(checkpoint, device)
        model._load_state_dict(checkpoint)
        return model

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "PatchSegmentationModel":
        """
        Create a model instance from checkpoint data.
        Must be implemented by subclasses.

        Args:
            checkpoint: The loaded checkpoint dictionary
            device: Device to create the model on

        Returns:
            An initialized model instance
        """
        raise NotImplementedError(
            "Subclasses must implement _create_from_checkpoint"
        )

    @staticmethod
    def _download_weights(
        url: str, save_path: Path, chunk_size: int = 1024
    ) -> None:
        """Download model weights with a progress bar (private method)."""
        response = requests.get(url, stream=True, verify=False)
        total_size = int(
            response.headers.get("content-length", 0)
        )  # Get total file size

        with (
            open(save_path, "wb") as file,
            tqdm(
                desc=f"Downloading {save_path.name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))

        logger.success(f"Download complete: {save_path}")

    def _load_state_dict(self, checkpoint: dict) -> None:
        """Load model weights from checkpoint (private method)."""
        self.model.load_state_dict(checkpoint["model_state"])

    def load(self, saved_path: Path, **kwargs) -> None:
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(saved_path, map_location=self.device)
        self._load_state_dict(checkpoint)

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """
        Return model-specific data for checkpoint saving.

        This method should be overridden by subclasses to provide
        model-specific parameters that need to be saved in checkpoints.

        Returns:
            Dictionary containing model-specific parameters
        """
        raise NotImplementedError(
            "Subclasses must implement _get_checkpoint_data"
        )

    def train(
        self,
        train_iterator: Iterator[tuple[np.ndarray, np.ndarray]] = None,
        val_iterator: Iterator[tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 1,
        n_steps: int = 100,
        val_steps: int = 10,
        out_dir: Path = None,
        LR: float = 0.001,
        scheduler_patience: int = 3,
        test_every: int = 0,
        test_params: TestParams = None,
        amp: bool = False,
        gradient_clipping: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Train the segmentation model.

        Args:
            train_iterator: Iterator yielding (image, mask) training batches
            val_iterator: Iterator yielding (image, mask) validation batches
            epochs: Number of training epochs
            n_steps: Number of training steps per epoch
            val_steps: Number of validation steps per epoch
            out_dir: Directory to save model checkpoints
            LR: Learning rate
            test_every: Test model every N epochs (0 to disable)
            test_params: Parameters for testing
            amp: Whether to use automatic mixed precision
            gradient_clipping: Gradient clipping value
            **kwargs: Additional keyword arguments
        """
        if out_dir is None:
            out_dir = Path.cwd() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.train_logger = TrainingLogger(
            out_dir / "train_log.json",
        )
        self.detailed_test_logger = DetailedTestLogger(
            out_dir / "test_log_detailed.json",
        )

        self.tester = None
        if test_params is not None and test_every > 0:
            self.tester = SegmDetailedTester(
                out_dir,
                classes=test_params.classes,
                detailed_logger=self.detailed_test_logger,
                void_pad=test_params.void_pad,
                void_border_width=test_params.void_border_width,
                vis_segmentation=test_params.vis_segmentation,
            )

        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=LR,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=scheduler_patience
        )
        grad_scaler = torch.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        best_miou = 0
        best_train_loss = float("inf")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"LR: {current_lr}")
            self.train_logger.log_learning_rate(epoch, current_lr)

            # ----- training -----
            self.model.train()
            epoch_loss = 0
            with tqdm(total=n_steps, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for i in range(n_steps):
                    img, mask = next(train_iterator)
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    img = img.permute(0, 3, 1, 2).contiguous()
                    img = (
                        img.to(
                            device=self.device,
                            dtype=torch.float32,
                        )
                        / 255.0
                    )
                    mask = mask.to(
                        device=self.device,
                        dtype=torch.long,
                    )
                    pred = self.model(img)
                    loss = self.compute_loss_with_auxiliary(
                        pred, mask, criterion, epoch, epochs
                    )
                    optimizer.zero_grad()
                    grad_scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clipping
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    epoch_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(**{"epoch loss": epoch_loss / (i + 1)})
            epoch_loss /= n_steps
            self.train_logger.log_loss(epoch, "train_loss", epoch_loss)
            logger.info(f"epoch loss: {epoch_loss}")

            # ----- validation -----
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for _ in tqdm(range(val_steps), "eval"):
                    img, mask = next(val_iterator)
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    img = img.permute(0, 3, 1, 2).contiguous()
                    img = (
                        img.to(
                            device=self.device,
                            dtype=torch.float32,
                        )
                        / 255.0
                    )
                    mask = mask.to(
                        device=self.device,
                        dtype=torch.long,
                    )
                    pred = self.model(img)
                    val_loss += self.compute_loss_with_auxiliary(
                        pred, mask, criterion, epoch, epochs
                    ).item()
                val_loss /= val_steps
                scheduler.step(val_loss)
                self.train_logger.log_loss(epoch, "val_loss", val_loss)
                logger.info(f"val loss: {val_loss}")

            # ----- save checkpoints -----
            ckpt_dir = out_dir / "models"
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

            ckpt = {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "optimizer_state": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "scheduler_state": scheduler.state_dict(),
                **self._get_checkpoint_data(),
            }

            torch.save(ckpt, ckpt_dir / "last_train_weights.pth")
            logger.info(f"Last train weights checkpoint {epoch} saved!")

            # Check if current losses are the best so far
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(ckpt, ckpt_dir / "best_train_loss_weights.pth")
                logger.info(f"Best train loss checkpoint {epoch} saved!")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, ckpt_dir / "best_val_loss_weights.pth")
                logger.info(f"Best val loss checkpoint {epoch} saved!")

            # ----- test on test set -----
            if self.tester is not None and epoch % test_every == 0:
                self.model.eval()
                metrics_full, metrics_void = self.tester.test_on_set(
                    test_params.img_mask_paths, self.predict_image, epoch=epoch
                )

                # Clean up old epoch visualization directories
                if (
                    test_params.vis_segmentation
                    and test_params.max_epoch_visualizations > 0
                ):
                    self._cleanup_epoch_visualizations(
                        out_dir, test_params.max_epoch_visualizations
                    )

                # Log dataset-level metrics to training logger
                self.train_logger.log_dataset_metrics(
                    epoch,
                    metrics_full,
                    void=False,
                    classes=test_params.classes,
                )
                self.train_logger.log_dataset_metrics(
                    epoch,
                    metrics_void,
                    void=True,
                    classes=test_params.classes,
                )

                logger.info(f"Metrics full \n{metrics_full}")
                logger.info(f"Metrics void \n{metrics_void}")

                # --- save best test mIoU checkpoint ---
                if metrics_full.mean_iou > best_miou:
                    best_miou = metrics_full.mean_iou
                    torch.save(
                        ckpt,
                        ckpt_dir / "best_test_miou_weights.pth",
                    )
                    logger.info(f"Best test mIoU checkpoint {epoch} saved!")

            # ----- generate plots after each epoch -----
            Plotter.plot_lrs(self.train_logger, out_dir=out_dir)
            Plotter.plot_losses(self.train_logger, out_dir=out_dir)
            colors = self.tester.classes.labels_to_colors_plt
            Plotter.plot_metrics(
                self.train_logger, void=False, out_dir=out_dir, colors=colors
            )
            Plotter.plot_metrics(
                self.train_logger, void=True, out_dir=out_dir, colors=colors
            )

        # ----- final message -----
        logger.info("Training completed! All plots have been generated.")

    def _cleanup_epoch_visualizations(
        self, out_dir: Path, max_epochs: int
    ) -> None:
        """
        Clean up old epoch visualization directories, keeping only the
        N most recent ones.

        Args:
            out_dir: Directory containing epoch subdirectories
            max_epochs: Maximum number of epoch directories to keep
        """
        if max_epochs <= 0:
            return

        # Find all epoch directories
        epoch_dirs = []
        for item in out_dir.iterdir():
            if item.is_dir() and item.name.startswith("epoch_"):
                try:
                    epoch_num = int(item.name.split("_")[1])
                    epoch_dirs.append((epoch_num, item))
                except (ValueError, IndexError):
                    # Skip directories that don't match epoch_N pattern
                    continue

        # Sort by epoch number and keep only the most recent ones
        if len(epoch_dirs) > max_epochs:
            epoch_dirs.sort(key=lambda x: x[0])  # Sort by epoch number
            # Keep only the last max_epochs
            dirs_to_remove = epoch_dirs[:-max_epochs]

            for epoch_num, dir_path in dirs_to_remove:
                try:
                    import shutil

                    shutil.rmtree(dir_path)
                    logger.info(
                        f"Removed old visualization directory: "
                        f"{dir_path.name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to remove directory {dir_path.name}: {e}"
                    )

    def predict_image(
        self,
        image: np.ndarray,
        return_logits: bool = True,
        pad_align: int = 16,
        patch_size_limit: int = 2048,
        patch_size: int = 2048,
        patch_stride: int | None = None,
    ) -> np.ndarray:
        """
        Predict segmentation for an image.

        Automatically chooses between full image or patch-based prediction
        based on image size relative to patch_size_limit.

        Args:
            image: Input image array (H, W, C)
            return_logits: If True, return raw logits; if False, return class
                indices
            pad_align: Padding alignment for model input
            patch_size_limit: Maximum image size for full prediction
            patch_size: Size of patches for large images
            patch_stride: Stride between patches (defaults to patch_size // 2)

        Returns:
            Segmentation prediction array
        """

        if image.ndim != 3:
            raise ValueError(
                f"Expected image with 3 dimensions (H, W, C), "
                f"got {image.ndim} dimensions."
            )

        if (
            image.shape[0] > patch_size_limit
            or image.shape[1] > patch_size_limit
        ):
            logger.warning(
                f"Image size {image.shape[:2]} exceeds limit "
                f"{patch_size_limit}x{patch_size_limit}. "
                "Using patched image prediction."
            )
            if patch_stride is None:
                patch_stride = patch_size // 2
            return self._predict_image_patched(
                image,
                return_logits=return_logits,
                pad_align=pad_align,
                patch_size=patch_size,
                patch_stride=patch_stride,
            )
        else:
            return self._predict_image_full(image, return_logits, pad_align)

    def _predict_image_patched(
        self,
        image: np.ndarray,
        patch_size: int,
        patch_stride: int,
        return_logits: bool = True,
        pad_align: int = 16,
    ) -> np.ndarray:

        from petroscope.segmentation.utils import (
            combine_from_patches,
            split_into_patches,
        )

        patches = split_into_patches(image, patch_size, patch_stride)

        preds = [
            self._predict_image_full(
                p, return_logits=True, pad_align=pad_align
            )
            for p in patches
        ]

        result = combine_from_patches(
            preds,
            patch_size,
            patch_stride,
            image.shape[:2],
        )
        if not return_logits:
            result = np.argmax(result, axis=-1).astype(np.uint8)

        return result

    def _predict_image_full(
        self,
        image: np.ndarray,
        return_logits: bool = True,
        pad_align: int = 16,
    ) -> np.ndarray:
        """
        Predicts the segmentation of a given image.

        Args:
            image: The input image to be segmented
            return_logits: Whether to return the raw logits instead of
                class indices

        Returns:
            Segmentation mask or logits
        """
        h, w = image.shape[:2]
        if h % pad_align != 0:
            pad_h = pad_align - (h % pad_align)
            image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)))
        if w % pad_align != 0:
            pad_w = pad_align - (w % pad_align)
            image = np.pad(image, ((0, 0), (0, pad_w), (0, 0)))

        self.model.eval()
        with torch.no_grad():
            p = (
                torch.from_numpy(image[np.newaxis, ...])
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            prediction = self.model(p)
            prediction = torch.sigmoid(prediction)
            if return_logits:
                prediction = (
                    prediction.squeeze().permute([1, 2, 0]).contiguous()
                )
            else:
                prediction = prediction.argmax(dim=1).squeeze()

            prediction = prediction.detach().cpu().numpy()

        prediction = prediction[:h, :w, ...]
        return prediction

    @property
    def n_params_str(self) -> str:
        """Get a string representation of the number of parameters."""
        from petroscope.utils.base import UnitsFormatter

        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Size of model: {UnitsFormatter.si(n)}"

    @property
    def n_params_str_detailed(self) -> int:
        """Get a detailed string representation of the model parameters."""
        from prettytable import PrettyTable

        def count_parameters(model) -> int:
            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
            print(table)
            print(f"Total Trainable Params: {total_params}")
            return total_params

        return count_parameters(self.model)

    def supports_auxiliary_loss(self) -> bool:
        """
        Check if this model supports auxiliary loss computation.
        Subclasses should override this to return True if they support
        auxiliary heads.
        """
        return False

    def get_auxiliary_loss_weight(
        self, epoch: int, total_epochs: int
    ) -> float:
        """
        Get the auxiliary loss weight for the current epoch.
        Weight typically decreases over training to focus on main output.

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs

        Returns:
            Auxiliary loss weight (0.0 to 1.0)
        """
        if epoch < total_epochs * 0.3:
            return 0.4  # Higher weight early in training
        elif epoch < total_epochs * 0.7:
            return 0.2  # Medium weight in middle
        else:
            return 0.1  # Lower weight later in training

    def compute_loss_with_auxiliary(
        self, pred, target, criterion, epoch: int = 0, total_epochs: int = 100
    ):
        """
        Compute loss including auxiliary loss if the model supports it.

        Args:
            pred: Model prediction(s) - single tensor or tuple (main, aux)
            target: Ground truth target
            criterion: Loss function
            epoch: Current epoch for auxiliary weight scheduling
            total_epochs: Total training epochs

        Returns:
            Combined loss tensor
        """
        is_aux_case = (
            isinstance(pred, tuple)
            and len(pred) == 2
            and self.supports_auxiliary_loss()
        )

        if is_aux_case:
            # Model returned (main_output, aux_output)
            main_pred, aux_pred = pred

            # Compute main loss
            main_loss = criterion(main_pred, target)

            # Compute auxiliary loss
            aux_loss = criterion(aux_pred, target)

            # Get auxiliary weight for current epoch
            aux_weight = self.get_auxiliary_loss_weight(epoch, total_epochs)

            # Combined loss
            total_loss = main_loss + aux_weight * aux_loss

            # Log auxiliary losses using existing log_loss method
            if hasattr(self, "train_logger") and self.train_logger is not None:
                self.train_logger.log_loss(
                    epoch, "main_loss", main_loss.item()
                )
                self.train_logger.log_loss(epoch, "aux_loss", aux_loss.item())
                self.train_logger.log_loss(epoch, "aux_weight", aux_weight)

            return total_loss
        else:
            # Standard single output case
            return criterion(pred, target)
