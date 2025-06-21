"""
Loss functions for semantic segmentation.

This module provides various loss functions that can be used for training
segmentation models, including standard losses and specialized losses for
handling class imbalance.
"""

from typing import Any

from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch, nn, F


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss with configurable parameters.

    This is a wrapper around PyTorch's CrossEntropyLoss to provide consistent
    interface with other loss functions.
    """

    def __init__(
        self,
        weight=None,
        ignore_index: int = 255,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Cross Entropy Loss.

        Args:
            weight: Manual rescaling weight given to each class
            ignore_index: Specifies a target value that is ignored
            reduction: Specifies the reduction to apply to the output
            label_smoothing: Amount of smoothing when computing the loss
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.

    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.
    (2017). Focal loss for dense object detection. ICCV, 2017.
    """

    def __init__(
        self,
        weight=None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            weight: Weighting factor for rare class (default: None)
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Specifies a target value that is ignored
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if isinstance(weight, (list, tuple)):
            self.weight = torch.tensor(weight, dtype=torch.float32)
        elif isinstance(weight, (int, float)):
            self.weight = torch.tensor([weight], dtype=torch.float32)

    def forward(self, pred, target):
        """
        Compute focal loss.

        Args:
            pred: Predictions of shape [N, C, H, W]
            target: Ground truth of shape [N, H, W]

        Returns:
            Computed loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction="none"
        )

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Create mask for ignored pixels
        mask = (target != self.ignore_index).float()

        # Apply class weighting
        if self.weight is not None:
            if self.weight.device != target.device:
                self.weight = self.weight.to(target.device)

            if len(self.weight) == 1:
                # Single weight value
                weight_t = self.weight[0] * torch.ones_like(
                    target, dtype=torch.float32
                )
            else:
                # Per-class weight values
                weight_t = self.weight.gather(0, target.view(-1))
                weight_t = weight_t.view_as(target)
        else:
            # Equal weights (1.0) for all classes when no weight is provided
            weight_t = torch.ones_like(target, dtype=torch.float32)

        # Apply focal weight formula (1-pt)^gamma with class weighting
        focal_weight = weight_t * (1 - pt) ** self.gamma

        # We've already computed the focal weight above

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            # Only compute mean over non-ignored pixels
            # Note: mask was already defined earlier
            return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif self.reduction == "sum":
            # Note: mask was already defined earlier
            return (focal_loss * mask).sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for semantic segmentation.

    Dice loss is particularly useful for segmentation tasks with small objects
    or high class imbalance.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = "mean",
        include_background: bool = True,
        weight=None,
    ):
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Specifies a target value that is ignored
            reduction: Specifies the reduction to apply to the output
            include_background: Whether to include background class in loss
            weight: Manual rescaling weight given to each class
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.include_background = include_background
        self.weight = weight

        if isinstance(weight, (list, tuple)):
            self.weight = torch.tensor(weight, dtype=torch.float32)
        elif isinstance(weight, (int, float)):
            self.weight = torch.tensor([weight], dtype=torch.float32)

    def forward(self, pred, target):
        """
        Compute Dice loss.

        Args:
            pred: Predictions of shape [N, C, H, W]
            target: Ground truth of shape [N, H, W]

        Returns:
            Computed loss value
        """
        # Convert predictions to probabilities
        pred_probs = F.softmax(pred, dim=1)

        # Create mask for ignored pixels first
        valid_mask = target != self.ignore_index

        # Replace ignored pixels with 0 for one-hot encoding
        target_masked = target.clone()
        target_masked[~valid_mask] = 0

        # Create one-hot encoding for target
        n_classes = pred.shape[1]
        target_one_hot = F.one_hot(target_masked, num_classes=n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Create mask for ignored pixels
        mask = valid_mask.float()
        mask = mask.unsqueeze(1).expand_as(pred_probs)

        # Apply mask
        pred_probs = pred_probs * mask
        target_one_hot = target_one_hot * mask

        # Compute Dice coefficient for each class
        dice_scores = []
        start_idx = 0 if self.include_background else 1

        for c in range(start_idx, n_classes):
            pred_c = pred_probs[:, c]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        # Convert to loss (1 - dice)
        dice_losses = 1.0 - torch.stack(dice_scores)

        # Apply weights if provided
        if self.weight is not None:
            if self.weight.device != pred.device:
                self.weight = self.weight.to(pred.device)
            # Adjust weights length to match the number of considered classes
            weight_adjusted = (
                self.weight[start_idx:] if start_idx > 0 else self.weight
            )
            if len(weight_adjusted) == len(dice_losses):
                dice_losses = dice_losses * weight_adjusted

        # Apply reduction
        if self.reduction == "mean":
            dice_loss = dice_losses.mean()
        elif self.reduction == "sum":
            dice_loss = dice_losses.sum()
        else:  # none
            dice_loss = dice_losses

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that can mix multiple loss types.

    This allows combining different loss functions with specified weights,
    e.g., combining CrossEntropy and Dice loss.
    """

    def __init__(
        self, losses: dict[str, dict[str, Any]], weights: dict[str, float]
    ):
        """
        Initialize Combined Loss.

        Args:
            losses: Dictionary mapping loss names to their configurations
            weights: Dictionary mapping loss names to their weights
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict()
        self.weights = weights

        for loss_name, loss_config in losses.items():
            # Get loss type without modifying the original config
            # (to support OmegaConf)
            loss_type = loss_config.get("type")

            # Create a copy of the config without the 'type' key
            config_params = {
                k: v for k, v in loss_config.items() if k != "type"
            }

            self.loss_functions[loss_name] = create_loss_function(
                loss_type, **config_params
            )

    def forward(self, pred, target):
        """
        Compute combined loss.

        Args:
            pred: Predictions of shape [N, C, H, W]
            target: Ground truth of shape [N, H, W]

        Returns:
            Computed combined loss value
        """
        total_loss = 0.0

        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.weights.get(loss_name, 1.0)
            loss_value = loss_fn(pred, target)
            total_loss += weight * loss_value

        return total_loss


def create_loss_function(loss_type: str, **kwargs):
    """
    Factory function to create loss functions by name.

    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional arguments for the loss function

    Returns:
        Initialized loss function
    """
    loss_type = loss_type.lower()

    # Map loss types to their classes
    loss_classes = {
        "crossentropy": CrossEntropyLoss,
        "focal": FocalLoss,
        "dice": DiceLoss,
        "combined": CombinedLoss,
    }

    if loss_type not in loss_classes:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss_classes[loss_type](**kwargs)


def get_class_weights(
    class_counts, method: str = "inverse", smooth: float = 1.0
):
    """
    Compute class weights for handling class imbalance.

    Args:
        class_counts: Number of pixels for each class
        method: Method for computing weights ('inverse', 'sqrt_inverse',
                'log_inverse', 'quadratic_inverse')
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Tensor of class weights
    """
    if isinstance(class_counts, list):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

    class_counts = class_counts.float()
    total_pixels = class_counts.sum()

    try:
        if method == "inverse":
            weights = total_pixels / (class_counts + smooth)
        elif method == "sqrt_inverse":
            weights = torch.sqrt(total_pixels / (class_counts + smooth))
        elif method == "log_inverse":
            weights = torch.log(total_pixels / (class_counts + smooth) + 1.0)
        elif method == "quadratic_inverse":
            weights = (total_pixels / (class_counts + smooth)) ** 2
        else:
            raise ValueError(f"Unsupported weight method: {method}")

        # Normalize weights so they sum to number of classes
        weights = weights / weights.sum() * len(weights)
        return weights
    except Exception as e:
        logger.error(f"Error calculating weights: {e}")
        raise


class LossManager:
    """
    Manages loss functions for segmentation tasks.

    This class handles the creation, configuration, and application of loss functions
    for segmentation models. It supports single loss functions as well as combined
    losses with weights. It also provides class weight calculation and application
    based on class distribution.
    """

    SUPPORTED_LOSS_TYPES = {"crossentropy", "focal", "dice", "combined"}

    SUPPORTED_WEIGHT_METHODS = {
        "inverse",
        "sqrt_inverse",
        "log_inverse",
        "quadratic_inverse",
    }

    def __init__(
        self,
        config: dict[str, Any],
        class_counts: dict[int, int] = None,
        device: str = None,
    ):
        """
        Initialize the loss manager.

        Args:
            config: Loss configuration dictionary (required)
            class_counts: Dictionary mapping class indices to pixel counts
            device: Device to place tensors on
        """
        if not config or "type" not in config:
            raise ValueError(
                "Loss configuration is required with a valid type"
            )

        # Store configuration
        self.config = config
        self.class_counts = class_counts
        self.device = device
        self.weights = None
        self.loss_fn = None

        # Process configuration and create loss function
        self._calculate_weights()
        self._create_loss_function()

    def _calculate_weights(self):
        """Calculate class weights based on configuration and class counts"""
        if "class_weights" not in self.config:
            return

        if not self.class_counts:
            return

        try:
            weight_config = self.config["class_weights"]

            if (
                not weight_config
                or "enabled" not in weight_config
                or not weight_config["enabled"]
            ):
                return

            method = weight_config.get("method")
            if not method or method not in self.SUPPORTED_WEIGHT_METHODS:
                raise ValueError(f"Unsupported weight method: {method}")

            smooth = weight_config.get("smooth_factor", 1.0)

            # Convert class_counts dict to ordered list
            class_indices = sorted(self.class_counts.keys())
            counts_list = [self.class_counts[idx] for idx in class_indices]

            # Calculate weights
            self.weights = get_class_weights(counts_list, method, smooth)

            if self.weights is not None and self.device:
                self.weights = self.weights.to(self.device)

            # Format weights to show 4 decimal places
            formatted_weights = None
            if self.weights is not None:
                formatted_weights = [f"{w:.4f}" for w in self.weights.tolist()]

            logger.info(
                f"Calculated class weights using '{method}' method: {formatted_weights}"
            )
        except Exception as e:
            logger.error(f"Error in _calculate_weights: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def _create_loss_function(self):
        """Create the loss function based on configuration"""
        loss_type = self.config["type"].lower()

        if loss_type not in self.SUPPORTED_LOSS_TYPES:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        if loss_type == "combined":
            self._create_combined_loss()
        else:
            self._create_single_loss(loss_type)

    def _create_single_loss(self, loss_type: str):
        """Create a single loss function"""
        # Get parameters for this loss type
        params = {}
        if loss_type in self.config:
            # Convert to regular dict to avoid OmegaConf errors when adding weight
            params = dict(self.config[loss_type])

        # Apply weights if available for all loss types
        if self.weights is not None:
            params["weight"] = self.weights

        # Create the loss function
        self.loss_fn = create_loss_function(loss_type, **params)

    def _create_combined_loss(self):
        """Create a combined loss function"""
        if "components" not in self.config or not self.config["components"]:
            raise ValueError("Combined loss requires at least one component")

        components = self.config["components"]

        # Prepare components and weights
        loss_dict = {}
        weights_dict = {}

        for i, component in enumerate(components):
            if "name" not in component:
                raise ValueError(f"Component {i} missing name field")

            name = component["name"]
            if name not in self.SUPPORTED_LOSS_TYPES:
                raise ValueError(f"Unsupported loss type: {name}")

            weight = 1.0
            if "weight" in component:
                weight = component["weight"]

            params = {}
            if "params" in component:
                # Convert to regular dict to avoid OmegaConf errors
                params = dict(component["params"])

            # Get base configuration for this loss type to use as defaults
            base_params = {}
            if name in self.config:
                # Convert to regular dict to avoid OmegaConf errors
                base_params = dict(self.config[name])

            # Merge parameters
            merged_params = {**base_params, **params}

            # Apply class weights if available to all loss types
            if self.weights is not None:
                merged_params["weight"] = self.weights

            # Store component configuration
            loss_dict[name] = {"type": name, **merged_params}
            weights_dict[name] = weight

        # Create the combined loss
        self.loss_fn = CombinedLoss(loss_dict, weights_dict)

    def __call__(self, pred, target):
        """
        Calculate the loss.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Loss value
        """
        return self.loss_fn(pred, target)

    @classmethod
    def from_config_and_dataset(
        cls, config: dict[str, Any], dataset, device: str = None
    ) -> "LossManager":
        """
        Create a LossManager from configuration and dataset.

        Args:
            config: Loss configuration dictionary (required)
            dataset: Dataset with get_class_pixel_counts method
            device: Device to place tensors on

        Returns:
            Initialized LossManager
        """
        class_counts = None

        # Get class pixel counts from dataset if available
        if hasattr(dataset, "get_class_pixel_counts"):
            class_counts = dataset.get_class_pixel_counts()

        return cls(config, class_counts, device)

    @classmethod
    def from_config_and_class_counts(
        cls,
        config: dict[str, Any],
        class_counts: dict[int, int],
        device: str = None,
    ) -> "LossManager":
        """
        Create a LossManager from configuration and class counts directly.

        Args:
            config: Loss configuration dictionary (required)
            class_counts: Dictionary mapping class indices to pixel counts
            device: Device to place tensors on

        Returns:
            Initialized LossManager
        """
        return cls(config, class_counts, device)
