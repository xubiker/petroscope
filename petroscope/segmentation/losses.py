"""
Loss functions for semantic segmentation.

This module provides various loss functions that can be used for training
segmentation models, including standard losses and specialized losses for
handling class imbalance.
"""

from typing import Dict, Any

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
        alpha=None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: None)
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Specifies a target value that is ignored
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

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

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)

            if len(self.alpha) == 1:
                # Single alpha value
                alpha_t = self.alpha[0]
            else:
                # Per-class alpha values
                alpha_t = self.alpha.gather(0, target.view(-1))
                alpha_t = alpha_t.view_as(target)

            # Only apply alpha where target is not ignored
            mask = (target != self.ignore_index).float()
            # Set alpha=1 for ignored pixels
            alpha_t = alpha_t * mask + (1 - mask)
        else:
            alpha_t = 1.0

        # Compute focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            # Only compute mean over non-ignored pixels
            mask = (target != self.ignore_index).float()
            return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif self.reduction == "sum":
            mask = (target != self.ignore_index).float()
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
    ):
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Specifies a target value that is ignored
            reduction: Specifies the reduction to apply to the output
            include_background: Whether to include background class in loss
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.include_background = include_background

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
        dice_loss = 1.0 - torch.stack(dice_scores).mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that can mix multiple loss types.

    This allows combining different loss functions with specified weights,
    e.g., combining CrossEntropy and Dice loss.
    """

    def __init__(
        self, losses: Dict[str, Dict[str, Any]], weights: Dict[str, float]
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

    Raises:
        ValueError: If loss_type is not supported
    """
    if loss_type is None:
        raise ValueError("Loss type cannot be None")

    loss_type = loss_type.lower()

    if loss_type == "crossentropy" or loss_type == "ce":
        return CrossEntropyLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_class_weights(
    class_counts, method: str = "inverse", smooth: float = 1.0
):
    """
    Compute class weights for handling class imbalance.

    Args:
        class_counts: Number of pixels for each class
        method: Method for computing weights ('inverse', 'sqrt_inverse',
                'log_inverse')
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Tensor of class weights
    """
    if isinstance(class_counts, list):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

    class_counts = class_counts.float()
    total_pixels = class_counts.sum()

    if method == "inverse":
        weights = total_pixels / (class_counts + smooth)
    elif method == "sqrt_inverse":
        weights = torch.sqrt(total_pixels / (class_counts + smooth))
    elif method == "log_inverse":
        weights = torch.log(total_pixels / (class_counts + smooth) + 1.0)
    else:
        raise ValueError(f"Unsupported weighting method: {method}")

    # Normalize weights so they sum to number of classes
    weights = weights / weights.sum() * len(weights)

    return weights
