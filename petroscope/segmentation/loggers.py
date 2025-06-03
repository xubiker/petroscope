from pathlib import Path
import json

from petroscope.segmentation.metrics import SegmMetrics


class TrainingLogger:
    """
    Lightweight logger for main training metrics (losses, learning rate,
    dataset-level metrics). This file remains small and easy to parse for
    training analysis and plotting.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.data = {"epochs": {}}

        # Create log file if it doesn't exist
        if not self.log_path.exists():
            self._save_to_file()
        else:
            self._load_from_file()

    def _load_from_file(self):
        """Load existing log data from file"""
        try:
            with open(self.log_path, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is corrupted, start fresh
            pass

    def _save_to_file(self):
        """Save current data to file"""
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def log_loss(self, epoch: int, loss_name: str, loss_value: float):
        """Log loss values"""
        self._ensure_epoch_structure(epoch)

        if "losses" not in self.data["epochs"][str(epoch)]:
            self.data["epochs"][str(epoch)]["losses"] = {}

        self.data["epochs"][str(epoch)]["losses"][loss_name] = loss_value
        self._save_to_file()

    def log_learning_rate(self, epoch: int, lr: float):
        """Log learning rate"""
        self._ensure_epoch_structure(epoch)
        self.data["epochs"][str(epoch)]["learning_rate"] = lr
        self._save_to_file()

    def log_dataset_metrics(
        self, epoch: int, metrics: SegmMetrics, void: bool = False
    ):
        """
        Log segmentation metrics for dataset-level evaluation only

        Args:
            epoch: Training epoch
            metrics: Segmentation metrics to log
            void: Whether void pixels were included in calculation
        """
        self._ensure_epoch_structure(epoch)

        if "metrics" not in self.data["epochs"][str(epoch)]:
            self.data["epochs"][str(epoch)]["metrics"] = {}

        # Store metrics with void context
        void_key = "void" if void else "full"

        if void_key not in self.data["epochs"][str(epoch)]["metrics"]:
            self.data["epochs"][str(epoch)]["metrics"][void_key] = {}

        # Dataset-level metrics with per-class IoUs
        self.data["epochs"][str(epoch)]["metrics"][void_key]["dataset"] = {
            "PA": metrics.acc.value,
            "hard": {"mIoU": metrics.mean_iou, "IoU_per_class": {}},
            "soft": {"mIoU": metrics.mean_iou_soft, "IoU_per_class": {}},
        }

        # Add hard IoU values per class
        for cl, iou in metrics.iou.items():
            self.data["epochs"][str(epoch)]["metrics"][void_key]["dataset"][
                "hard"
            ]["IoU_per_class"][cl] = iou.value

        # Add soft IoU values per class
        for cl, iou_soft in metrics.iou_soft.items():
            self.data["epochs"][str(epoch)]["metrics"][void_key]["dataset"][
                "soft"
            ]["IoU_per_class"][cl] = iou_soft.value

        self._save_to_file()

    def _ensure_epoch_structure(self, epoch: int):
        """Ensure the epoch structure exists in data"""
        epoch_str = str(epoch)
        if epoch_str not in self.data["epochs"]:
            self.data["epochs"][epoch_str] = {}

    def get_losses(self, loss_name: str = None) -> dict[int, float]:
        """Retrieve loss data for plotting"""
        losses = {}
        for epoch, epoch_data in self.data["epochs"].items():
            if "losses" in epoch_data:
                if loss_name:
                    if loss_name in epoch_data["losses"]:
                        losses[int(epoch)] = epoch_data["losses"][loss_name]
                else:
                    losses[int(epoch)] = epoch_data["losses"]
        return losses

    def get_metrics(self, void: bool = False) -> dict[int, dict[str, float]]:
        """Retrieve dataset-level metrics data for plotting"""
        void_key = "void" if void else "full"
        metrics_data = {}

        for epoch, epoch_data in self.data["epochs"].items():
            if "metrics" in epoch_data:
                if void_key in epoch_data["metrics"]:
                    if "dataset" in epoch_data["metrics"][void_key]:
                        metrics_data[int(epoch)] = epoch_data["metrics"][
                            void_key
                        ]["dataset"]

        return metrics_data

    def get_lr(self) -> dict[int, float]:
        """Retrieve learning rate data for plotting"""
        lr_data = {}
        for epoch, epoch_data in self.data["epochs"].items():
            if "learning_rate" in epoch_data:
                lr_data[int(epoch)] = epoch_data["learning_rate"]
        return lr_data


class DetailedTestLogger:
    """
    Detailed logger for per-image metrics. This file can grow large but is
    only accessed when debugging specific image predictions.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.data = {"epochs": {}}

        # Create log file if it doesn't exist
        if not self.log_path.exists():
            self._save_to_file()
        else:
            self._load_from_file()

    def _load_from_file(self):
        """Load existing log data from file"""
        try:
            with open(self.log_path, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is corrupted, start fresh
            pass

    def _save_to_file(self):
        """Save current data to file"""
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def log_image_metrics(
        self,
        epoch: int,
        image_name: str,
        metrics: SegmMetrics,
        void: bool = False,
    ):
        """
        Log segmentation metrics for individual images with per-class IoUs

        Args:
            epoch: Training epoch
            image_name: Name of the image
            metrics: Segmentation metrics for this image
            void: Whether void pixels were included in calculation
        """
        self._ensure_epoch_structure(epoch)

        void_key = "void" if void else "full"

        if void_key not in self.data["epochs"][str(epoch)]:
            self.data["epochs"][str(epoch)][void_key] = {}

        # Store per-image metrics with per-class IoUs
        image_data = {
            "PA": metrics.acc.value,
            "hard": {"mIoU": metrics.mean_iou, "IoU_per_class": {}},
            "soft": {"mIoU": metrics.mean_iou_soft, "IoU_per_class": {}},
        }

        # Add hard IoU values per class
        for cl, iou in metrics.iou.items():
            image_data["hard"]["IoU_per_class"][cl] = iou.value

        # Add soft IoU values per class
        for cl, iou_soft in metrics.iou_soft.items():
            image_data["soft"]["IoU_per_class"][cl] = iou_soft.value

        self.data["epochs"][str(epoch)][void_key][image_name] = image_data
        self._save_to_file()

    def _ensure_epoch_structure(self, epoch: int):
        """Ensure the epoch structure exists in data"""
        epoch_str = str(epoch)
        if epoch_str not in self.data["epochs"]:
            self.data["epochs"][epoch_str] = {}

    def get_image_metrics(
        self, void: bool = False, epoch: int = None
    ) -> dict[int, dict[str, dict]]:
        """
        Retrieve per-image metrics data

        Args:
            void: Whether to get void or full metrics
            epoch: Specific epoch (if None, returns all epochs)

        Returns:
            Dictionary with structure: {epoch: {image_name: metrics_data}}
        """
        void_key = "void" if void else "full"
        image_metrics = {}

        epochs_to_check = (
            [str(epoch)] if epoch is not None else self.data["epochs"].keys()
        )

        for epoch_str in epochs_to_check:
            if epoch_str in self.data["epochs"]:
                epoch_data = self.data["epochs"][epoch_str]
                if void_key in epoch_data:
                    image_metrics[int(epoch_str)] = epoch_data[void_key]

        return image_metrics
