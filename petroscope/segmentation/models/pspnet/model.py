from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np
import requests
from tqdm import tqdm

from petroscope.segmentation.eval import SegmDetailedTester
from petroscope.segmentation.model import GeoSegmModel
from petroscope.segmentation.utils.data import ClassSet

# import torch-sensitive modules (satisfies Pylance and Flake8)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

from petroscope.utils import logger
from petroscope.utils.lazy_imports import nn, optim, torch  # noqa


class PSPNetTorch(GeoSegmModel):

    MODEL_REGISTRY: dict[str, str] = {
        "s1_x05_36_10": "https://www.xubiker.online/petroscope/segmentation_weights/pspnet_s1_x05_36_10.pth",
    }

    CACHE_DIR = Path.home() / ".petroscope" / "models"

    @dataclass
    class TestParams:
        classes: ClassSet
        img_mask_paths: Iterable[tuple[str, str]]
        void_pad: int
        void_border_width: int
        vis_plots: bool
        vis_segmentation: bool

    def __init__(
        self, n_classes: int, backbone: str, dilated: bool, device: str
    ) -> None:
        super().__init__()

        from petroscope.segmentation.models.pspnet.nn import PSPNet

        self.device = device
        self.model = PSPNet(
            n_classes=n_classes, dilated=dilated, backbone=backbone
        ).to(self.device)

    @staticmethod
    def download_weights(url: str, save_path: Path, chunk_size: int = 1024):
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

    @classmethod
    def trained(cls, weights_name: str, device: str) -> "PSPNetTorch":
        """Load a trained model from the registry, restoring hyperparameters automatically."""
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
        if not weights_path.exists():
            logger.info(f"Downloading weights for {weights_name}...")
            cls.download_weights(weights_url, weights_path)

        checkpoint = torch.load(weights_path, map_location=device)

        # Extract architecture hyperparameters from checkpoint
        n_classes = checkpoint["n_classes"]
        dilated = checkpoint["dilated"]
        backbone = checkpoint["backbone"]

        # Create the model with stored hyperparameters
        model = cls(
            n_classes=n_classes,
            dilated=dilated,
            backbone=backbone,
            device=device,
        )
        model.load(weights_path)
        return model

    def load(self, saved_path: Path, **kwargs) -> None:
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(saved_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def train(
        self,
        img_mask_paths: Iterable[tuple[Path, Path]],
        train_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        val_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        epochs: int,
        n_steps: int,
        val_steps: int,
        out_dir: Path,
        LR: float,
        test_every: int = 0,
        test_params: TestParams = None,
        amp: bool = False,
        gradient_clipping: float = 1.0,
    ) -> None:

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

            checkpoint = {
                "model_state": self.model.state_dict(),  # model weights
                "n_classes": self.model.n_classes,  # number of classes
                "backbone": self.model.backbone,  # backbone
                "dilated": self.model.dilated,
                "epoch": epoch,  # current epoch
                "optimizer_state": optimizer.state_dict(),  # optimizer state (optional)
                "train_loss": epoch_loss,  # Track training loss
                "val_loss": val_loss,  # Track validation loss
                "scheduler_state": scheduler.state_dict(),  # Save LR scheduler state
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
        patch_overlay: int | float,
    ) -> np.ndarray:
        from petroscope.segmentation.utils.data import (
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
        retutn_logits: bool = True,
    ) -> np.ndarray:
        """
        Predicts the segmentation of a given image.

        Args:
            image (ndarray): The input image to be segmented.
            retutn_logits (bool, optional): Whether to return the raw logits
            instead of the segmented class indices. Defaults to False.

        Returns:
            ndarray: The segmented image, either as class indices or raw logits
            depending on the value of `retutn_logits`.
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
            if retutn_logits:
                prediction = prediction.squeeze().permute([1, 2, 0])
            else:
                prediction = prediction.argmax(dim=1).squeeze()

            prediction = prediction.detach().cpu().numpy()

        prediction = prediction[:h, :w, ...]
        return prediction

    def predict_image_with_shift(
        self, image: np.ndarray, shift: int = 192
    ) -> np.ndarray:
        h, w = image.shape[:2]
        q = 16
        if h % q != 0:
            pad_h = q - (h % q)
            image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)))
        if w % q != 0:
            pad_w = q - (w % q)
            image = np.pad(image, ((0, 0), (0, pad_w), (0, 0)))

        shifts = ((0, 0), (0, shift), (shift, 0), (shift, shift))

        self.model.eval()
        with torch.no_grad():
            p = (
                torch.from_numpy(image[np.newaxis, ...])
                .permute(0, 3, 1, 2)
                .to(self.device)
            )

            preds = []
            for shy, shx in shifts:
                pred = (
                    torch.sigmoid(self.model(p[:, :, shy:, shx:]))
                    .cpu()
                    .numpy()
                )
                pred = np.pad(pred, ((0, 0), (0, 0), (shy, 0), (shx, 0)))
                preds.append(pred)

            pred_res = np.sum(preds, axis=0).argmax(axis=1).squeeze()
            pred_res = pred_res[:h, :w]
            return pred_res

    @property
    def n_params_str(self):
        from petroscope.segmentation.utils.base import UnitsFormatter

        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Size of model: {UnitsFormatter.si(n)}"

    @property
    def n_params_str_detailed(self):

        from prettytable import PrettyTable

        def count_parameters(model):
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
