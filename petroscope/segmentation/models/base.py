from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import requests
from tqdm import tqdm

from petroscope.segmentation.classes import ClassSet
from petroscope.segmentation.eval import SegmDetailedTester
from petroscope.segmentation.models.abstract import GeoSegmModel

# import torch-sensitive modules (satisfies Pylance and Flake8)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

from petroscope.utils import logger
from petroscope.utils.lazy_imports import nn, optim, torch  # noqa


class PatchSegmentationModel(GeoSegmModel):
    """
    Base class for patch-based segmentation models with common training pipeline.

    This class implements common functionality shared between different segmentation
    models like ResUNet, PSPNet, and UPerNet that are trained on image patches.
    It provides methods for training, prediction, and model evaluation.

    Subclasses should implement:
    - MODEL_REGISTRY: A dictionary mapping model names to weight URLs
    - __init__: Initialize the specific model architecture
    - create_from_checkpoint: Class method to create a model instance from checkpoint data
    - get_checkpoint_data: Return model-specific data for checkpoint saving
    """

    MODEL_REGISTRY: Dict[str, str] = {}  # Maps model names to weight URLs
    CACHE_DIR = Path.home() / ".petroscope" / "models"

    @dataclass
    class TestParams:
        classes: ClassSet
        img_mask_paths: Iterable[Tuple[str, str]]
        void_pad: int
        void_border_width: int
        vis_plots: bool
        vis_segmentation: bool

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
        self.tester: Optional[SegmDetailedTester] = None

    @classmethod
    def trained(
        cls, weights_name: str, device: str, force_download: bool = False
    ) -> "PatchSegmentationModel":
        """
        Generic method to load a trained model from a registry.

        Args:
            weights_name: Name of the weights in the model registry
            device: Device to load the model on
            force_download: Whether to force download even if weights exist locally

        Returns:
            Initialized and loaded model
        """
        if weights_name not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model version '{weights_name}'. Available: {list(cls.MODEL_REGISTRY.keys())}"
            )

        weights_url = cls.MODEL_REGISTRY[weights_name]
        weights_path = (
            Path.home() / ".cache" / "petroscope" / f"{weights_name}.pth"
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # Download if not available
        if not weights_path.exists() or force_download:
            logger.info(f"Downloading weights for {weights_name}...")
            cls.download_weights(weights_url, weights_path)

        checkpoint = torch.load(weights_path, map_location=device)

        # Create model using subclass-specific method
        model = cls.create_from_checkpoint(checkpoint, device)
        model.load(weights_path)
        return model

    @classmethod
    def create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "PatchSegmentationModel":
        """
        Create a model instance from checkpoint data.

        This method must be implemented by subclasses to extract
        model-specific parameters and create an appropriate instance.

        Args:
            checkpoint: The loaded checkpoint dictionary
            device: Device to create the model on

        Returns:
            An initialized model instance
        """
        raise NotImplementedError(
            "Subclasses must implement create_from_checkpoint"
        )

    @staticmethod
    def download_weights(
        url: str, save_path: Path, chunk_size: int = 1024
    ) -> None:
        """Download model weights with a progress bar."""
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

    def load(self, saved_path: Path, **kwargs) -> None:
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(saved_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Return model-specific data for checkpoint saving.

        This method should be overridden by subclasses to provide
        model-specific parameters that need to be saved in checkpoints.

        Returns:
            Dictionary containing model-specific parameters
        """
        raise NotImplementedError(
            "Subclasses must implement get_checkpoint_data"
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
        self.tester = None
        if test_params is not None and test_every > 0:
            self.tester = SegmDetailedTester(
                out_dir,
                classes=test_params.classes,
                void_pad=test_params.void_pad,
                void_border_width=test_params.void_border_width,
                vis_segmentation=test_params.vis_segmentation,
                vis_plots=test_params.vis_plots,
            )

        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=LR,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=5
        )
        grad_scaler = torch.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        epoch_losses = []

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"LR: {optimizer.param_groups[0]['lr']}")
            self.model.train()
            epoch_loss = 0
            with tqdm(total=n_steps, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for i in range(n_steps):
                    img, mask = next(train_iterator)
                    img = torch.tensor(img)
                    mask = torch.tensor(mask)
                    img = img.to(
                        device=self.device,
                        dtype=torch.float32,
                    ).permute(0, 3, 1, 2)
                    mask = mask.to(
                        device=self.device,
                        dtype=torch.long,
                    )
                    pred = self.model(img)
                    loss = criterion(pred, mask)
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
            epoch_losses.append(epoch_loss)
            logger.info(f"epoch loss: {epoch_loss}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for _ in tqdm(range(val_steps), "eval"):
                    img, mask = next(val_iterator)
                    img = torch.tensor(img)
                    mask = torch.tensor(mask)
                    img = img.to(
                        device=self.device,
                        dtype=torch.float32,
                    ).permute(0, 3, 1, 2)
                    mask = mask.to(
                        device=self.device,
                        dtype=torch.long,
                    )
                    pred = self.model(img)
                    val_loss += criterion(pred, mask).item() / val_steps
                scheduler.step(val_loss)
                logger.info(f"val loss: {val_loss}")

            # save checkpoint:
            checkpoint_dir = out_dir / "models"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

            logger.info("Saving model...")

            # Get model-specific checkpoint data
            checkpoint_data = self.get_checkpoint_data()

            # Add common checkpoint data
            checkpoint = {
                "model_state": self.model.state_dict(),  # model weights
                "epoch": epoch,  # current epoch
                "optimizer_state": optimizer.state_dict(),  # optimizer state
                "train_loss": epoch_loss,  # Track training loss
                "val_loss": val_loss,  # Track validation loss
                "scheduler_state": scheduler.state_dict(),  # Save LR scheduler state
                **checkpoint_data,  # Add model-specific data
            }

            torch.save(
                checkpoint, checkpoint_dir / f"weights_epoch_{epoch}.pth"
            )

            if epoch_loss <= min(epoch_losses):
                torch.save(checkpoint, checkpoint_dir / "weights_best.pth")
                logger.info(f"Best checkpoint {epoch} saved!")

            # test model
            if self.tester is not None and epoch % test_every == 0:
                self.model.eval()
                metrics, metrics_void = self.tester.test_on_set(
                    test_params.img_mask_paths,
                    self.predict_image,
                    description=f"epoch {epoch}",
                )
                logger.info(f"Metrics \n{metrics}")
                logger.info(f"Metrics void \n{metrics_void}")

    def predict_image_per_patches(
        self,
        image: np.ndarray,
        patch_s: int,
        batch_s: int,
        conv_pad: int,
        patch_overlay: Union[int, float],
    ) -> np.ndarray:
        """
        Predict segmentation by processing image in patches.

        Args:
            image: Input image
            patch_s: Patch size
            batch_s: Batch size
            conv_pad: Convolution padding
            patch_overlay: Patch overlay factor

        Returns:
            Segmentation mask
        """
        from petroscope.segmentation.utils import (
            combine_from_patches,
            split_into_patches,
        )

        patches = split_into_patches(image, patch_s, conv_pad, patch_overlay)
        init_patch_len = len(patches)

        while len(patches) % batch_s != 0:
            patches.append(patches[-1])
        pred_patches = []

        self.model.eval()
        with torch.no_grad():

            for i in range(0, len(patches), batch_s):
                batch = np.stack(patches[i : i + batch_s])
                batch = (
                    torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
                )
                prediction = self.model(batch)
                prediction = torch.sigmoid(prediction).argmax(dim=1)
                prediction = prediction.detach().cpu().numpy()
                for x in prediction:
                    pred_patches.append(x)

        pred_patches = pred_patches[:init_patch_len]
        result = combine_from_patches(
            pred_patches,
            patch_s,
            conv_pad,
            patch_overlay,
            image.shape[:2],
        )
        return result

    def predict_image(
        self,
        image: np.ndarray,
        return_logits: bool = True,
    ) -> np.ndarray:
        """
        Predicts the segmentation of a given image.

        Args:
            image: The input image to be segmented
            return_logits: Whether to return the raw logits instead of class indices

        Returns:
            Segmentation mask or logits
        """
        h, w = image.shape[:2]
        q = 16
        if h % q != 0:
            pad_h = q - (h % q)
            image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)))
        if w % q != 0:
            pad_w = q - (w % q)
            image = np.pad(image, ((0, 0), (0, pad_w), (0, 0)))

        self.model.eval()
        with torch.no_grad():
            p = (
                torch.from_numpy(image[np.newaxis, ...])
                .permute(0, 3, 1, 2)
                .to(self.device)
            )
            prediction = self.model(p)
            prediction = torch.sigmoid(prediction)
            if return_logits:
                prediction = prediction.squeeze().permute([1, 2, 0])
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
