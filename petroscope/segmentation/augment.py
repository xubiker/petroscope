import numpy as np
import albumentations as A
import cv2


class PrimaryAugmentor:
    """
    Albumentations‐based augmentor that performs, in order:
      1. random scale (zoom)
      2. random rotation
      3. horizontal + vertical flips
      4. brightness change
      5. color change
      6. center crop to final patch size

    Works for both image-only and (image, mask) inputs;
    masks always get nearest‐neighbor interpolation.
    """

    def __init__(
        self,
        patch_size: int,
        scale_limit: float = None,
        rot_angle_limit: float = None,
        brightness_shift: int = None,
        color_shift: float | None = None,
        seed: int = None,
    ) -> None:
        """
        Initializes the augmentor.

        Args:
            patch_size: desired size of the output patch
            scale_limit: maximum scale multiplier
            rot_angle_limit: maximum rotation angle in degrees
            min_rot_angle: minimum rotation angle in degrees
            brightness_shift: range of brightness change per channel (0-1)
            color_shift: range of color change per channel (0-1)
            seed: random seed for reproducible augmentations
        """
        # Store params and precompute ranges
        self.patch_size_trg = patch_size
        self.scale_limit = scale_limit
        self.rot_angle_limit = rot_angle_limit
        self.brightness_shift = brightness_shift
        self.color_shift = color_shift
        self.seed = seed

        # Precompute ranges for transformations
        self.scale_range = (
            (1 / (1 + self.scale_limit), 1 + self.scale_limit)
            if self.scale_limit
            else None
        )
        self.rotation_range = (
            (-self.rot_angle_limit, self.rot_angle_limit)
            if self.rot_angle_limit
            else None
        )

        # Calculate intermediate sizes to prevent clipping during transforms
        self.patch_size_int = patch_size
        if self.rot_angle_limit is not None:
            alpha = self.rot_angle_limit / 180 * np.pi
            expand = np.sin(alpha) + np.cos(alpha)
            self.patch_size_int = int(np.ceil(patch_size * expand))

        self.patch_size_src = self.patch_size_int
        if self.scale_limit is not None:
            self.patch_size_src = int(
                np.ceil(self.patch_size_int * (1 + self.scale_limit))
            )

        # Build Albumentations pipeline with proper transforms
        self.transform = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """
        Build the Albumentations transformation pipeline.

        Returns:
            Composed transform pipeline
        """
        tfms = []

        # 1) Combined scale and rotation using a single Affine transform
        if self.scale_range is not None or self.rotation_range is not None:
            # Default values if either parameter is None
            scale = self.scale_range if self.scale_range else (1.0, 1.0)
            rotate = self.rotation_range if self.rotation_range else 0

            # Create a single affine transform for both operations
            affine_transform = A.Affine(
                scale=scale,
                rotate=rotate,
                translate_percent=0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fit_output=False,
                p=1.0,
            )
            tfms.append(affine_transform)

        # 2) Flips
        hflip_transform = A.HorizontalFlip(p=0.5)
        vflip_transform = A.VerticalFlip(p=0.5)
        tfms.append(hflip_transform)
        tfms.append(vflip_transform)

        # 3) Brightness change
        if self.brightness_shift is not None:
            brightness_transform = A.RandomBrightnessContrast(
                brightness_limit=self.brightness_shift,
                contrast_limit=0,
                p=1.0,
            )
            tfms.append(brightness_transform)

        # 3) Color change
        if self.color_shift is not None:
            shift = int(255 * self.color_shift)
            color_transform = A.RGBShift(
                r_shift_limit=shift,
                g_shift_limit=shift,
                b_shift_limit=shift,
                p=1.0,
            )
            tfms.append(color_transform)

        # 4) Final center crop
        if self.patch_size_int != self.patch_size_trg:
            crop_transform = A.CenterCrop(
                height=self.patch_size_trg,
                width=self.patch_size_trg,
                p=1.0,
            )
            tfms.append(crop_transform)

        # Regular Compose is more efficient, now with seed parameter
        return A.Compose(
            tfms,
            additional_targets={"mask": "mask"},
            seed=self.seed,
        )

    def augment(
        self,
        img: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Apply the augmentation pipeline.

        Args:
            img: H×W×3 uint8 array (image)
            mask: H×W single‐channel uint8 array (optional mask)

        Returns:
            If mask is given: (augmented_img, augmented_mask)
            else: augmented_img
        """
        # Ensure input image is a 3D array with expected shape

        if (
            img.ndim != 3
            or img.shape[0] != img.shape[1]
            or img.dtype != np.uint8
        ):
            raise ValueError(
                "Input image must be square uint8 with 3 channels"
            )

        if mask is not None:
            if (
                mask.ndim != 2
                or mask.shape[0] != mask.shape[1]
                or mask.dtype != np.uint8
                or mask.shape[0] != img.shape[0]
            ):
                raise ValueError(
                    "Input mask must be square uint8 with 1 channel "
                    "and same size as image"
                )

        # Prepare data
        data = {"image": img}
        if mask is not None:
            data["mask"] = mask

        # Apply transform
        transformed = self.transform(**data)

        # Extract and return the transformed data
        augmented_img = transformed["image"]

        if mask is not None:
            augmented_mask = transformed["mask"]
            return augmented_img, augmented_mask
        else:
            return augmented_img

    def patch_sizes(self) -> tuple[int, int, int]:
        """
        Returns the patch sizes of the augmentor at different stages.

        Returns:
            Tuple of patch sizes (source, intermediate, target)
        """
        return self.patch_size_src, self.patch_size_int, self.patch_size_trg
