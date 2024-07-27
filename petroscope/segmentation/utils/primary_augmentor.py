import numpy as np
from PIL import Image


class PrimaryAugmentor:
    """
    Primary augmentor class that performs the following augmentations:
    - rescaling
    - rotation
    - flipping

    The augmentations are performed in the following sequence:
    scale -> rotate -> flip
    """

    def __init__(
        self,
        patch_size: int,
        max_scale: float | None = None,
        max_rot_angle: float | None = None,
        min_rot_angle: float = 1.0,
    ) -> None:
        """
        Initializes the augmentor.

        Args:
            patch_size: desired size of the output patch
            max_scale: maximum scale multiplier
            max_rot_angle: maximum rotation angle
            min_rot_angle: minimum rotation angle
        """

        self.max_scale = max_scale
        self.max_rot_angle = max_rot_angle
        self.min_rot_angle = min_rot_angle

        # patch size sequence:
        # source s -> {rescale op} -> intermediate s -> {rotate op} -> target s
        self.patch_size_src = patch_size
        self.patch_size_int = patch_size
        self.patch_size_trg = patch_size
        if max_rot_angle is not None:
            alpha = max_rot_angle / 180 * np.pi
            enlarge_coeff = np.sin(alpha) + np.cos(alpha)
            self.patch_size_src = int(
                np.ceil(self.patch_size_src * enlarge_coeff)
            )
        if max_scale is not None:
            enlarge_coeff = 1 + max_scale
            self.patch_size_int = int(
                np.ceil(self.patch_size_int * enlarge_coeff)
            )
            self.patch_size_src = int(
                np.ceil(self.patch_size_src * enlarge_coeff)
            )

    def _central_crop(self, a: np.ndarray, size: int) -> np.ndarray:
        """
        Crops the input array to a central square.

        Args:
            a: input array
            size: size of the central square

        Returns:
            cropped array
        """

        d = (a.shape[0] - size) // 2
        a = a[d : d + size, d : d + size, ...]
        return a

    def augment(
        self,
        img: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str] | tuple[np.ndarray, str]:
        """
        Applies the augmentations to the input image and mask
        (if it is provided).

        Args:
            img: input image
            mask: input mask (optional)

        Returns:
            augmented image, mask (optional) and applied augmentation code.
        """
        # Ensure input image is a 3D array with square shape
        assert (
            img.ndim == 3
            and img.shape[0] == img.shape[1] == self.patch_size_src
        )
        if mask is not None:
            assert (
                mask.ndim == 2
                and mask.shape[0] == mask.shape[1] == self.patch_size_src
            )

        # List to store the applied augmentations
        applied_augmentations = []

        # Initialize PIL images
        img_pil = None
        mask_pil = None

        # Perform rescale augmentation
        if self.max_scale is not None:
            sign = np.random.choice([-1, 1])
            rescale_mult = 1 + sign * np.random.rand() * self.max_scale
            applied_augmentations.append(f"z{rescale_mult:.2f}")
            s = int(self.patch_size_int * rescale_mult)

            img = self._central_crop(img, s)
            if mask is not None:
                mask = self._central_crop(mask, s)

            img_pil = Image.fromarray(img).resize(
                (self.patch_size_int, self.patch_size_int),
                resample=Image.BILINEAR,
            )
            if mask is not None:
                mask_pil = Image.fromarray(mask).resize(
                    (self.patch_size_int, self.patch_size_int),
                    resample=Image.NEAREST,
                )

        # Perform rotation augmentation
        if self.max_rot_angle is not None:
            alpha = (
                np.random.rand() * (self.max_rot_angle - self.min_rot_angle)
                + self.min_rot_angle
            )
            applied_augmentations.append(f"rot{alpha:.2f}")

            if img_pil is None:
                img_pil = Image.fromarray(img)
                if mask is not None:
                    mask_pil = Image.fromarray(mask)

            img_pil = img_pil.rotate(
                angle=alpha, expand=False, resample=Image.BILINEAR
            )
            if mask is not None:
                mask_pil = mask_pil.rotate(
                    angle=alpha, expand=False, resample=Image.NEAREST
                )

        # Convert PIL images back to numpy arrays
        if img_pil is not None:
            img = np.array(img_pil)
        if mask_pil is not None and mask is not None:
            mask = np.array(mask_pil)

        # Perform central crop to target size
        if self.max_scale is not None or self.max_rot_angle is not None:
            img = self._central_crop(img, self.patch_size_trg)
            if mask is not None:
                mask = self._central_crop(mask, self.patch_size_trg)

        if np.random.rand() > 0.5:
            applied_augmentations.append("hfl")
            img = np.fliplr(img)
            if mask is not None:
                mask = np.fliplr(mask)
        if np.random.rand() > 0.5:
            applied_augmentations.append("vfl")
            img = np.flipud(img)
            if mask is not None:
                mask = np.flipud(mask)

        # Generate augmentation code
        augm_code = "_".join(applied_augmentations)

        if mask is not None:
            return img, mask, augm_code
        else:
            return img, augm_code

    def patch_sizes(self) -> tuple[int]:
        """
        Returns the patch sizes of the augmentor.

        Returns:
            tuple of patch sizes during all steps of augmentation.
        """

        return self.patch_size_src, self.patch_size_int, self.patch_size_trg
