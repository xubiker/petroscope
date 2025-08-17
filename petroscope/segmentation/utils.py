import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from petroscope.utils import logger


def avg_pool_2d(mat: np.ndarray, kernel_size: int = 4) -> np.ndarray:
    """Performs a 2D average pooling operation on a given matrix.

    Args:
        mat (np.ndarray): The input matrix.
        kernel_size (int, optional): The size of the kernel. Defaults to 4.

    Returns:
        np.ndarray: The result of the average pooling operation.
    """
    assert mat.ndim == 2
    M, N = mat.shape

    # Shape of kernel
    K = kernel_size
    L = kernel_size

    # Dividing the image size by kernel size
    MK = M // K
    NL = N // L

    # Creating a pool
    res = mat[: MK * K, : NL * L].reshape(MK, K, NL, L).mean(axis=(1, 3))

    return res


def to_categorical(x: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converts a class mask to a one-hot encoded label mask.

    Args:
        x: A numpy array of shape (H, W) where each value is an integer
            representing a class label.
        n_classes: An integer representing the total number of classes.

    Returns:
        A numpy array of shape (H, W, n_classes) where each value is
        either 0 or 1,
        indicating the presence or absence of a class at that location.
    """
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if x.dtype != np.uint8:
        raise ValueError("Input must be of type uint8")
    input_shape = x.shape
    x = x.reshape(-1)
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, n_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_image(path: Path, normalize=False, to_float=False) -> np.ndarray:
    """
    Load an image from the given file path and optionally normalize it.

    Args:
        path (Path): The path to the image file.
        normalize (bool, optional): Whether to normalize the image.
        Defaults to False. If True, the image is converted to float32 and
        normalized to the range [0, 1]
        to_float (bool, optional): Whether to convert the image to float32.
        Defaults to False.

    Returns:
        np.ndarray: The loaded image as an RGB numpy array. Defaults to
        uint8.
    """
    import cv2

    # copy needed to avoid side effects while transferring arrays to pytorch
    img = cv2.imread(str(path))[:, :, ::-1].copy()  # BGR to RGB
    if normalize:
        return img.astype(np.float32) / 255
    if to_float:
        return img.astype(np.float32)
    return img


def load_mask(
    path: Path,
    max_classes: int | None = None,
    one_hot=False,
    to_float=False,
) -> np.ndarray:
    """
    Load a mask from the given file path and preprocess it.

    Args:
        path (Path): The path to the mask file.
        max_classes (int | None): Maximum number of classes for one-hot
        encoding. Required if one_hot=True.
        one_hot (bool, optional): Whether to convert the mask to one-hot
        encoding. Defaults to False.
        to_float (bool, optional): Whether to convert the mask to float32.
        Defaults to False.

    Returns:
        np.ndarray: The loaded and preprocessed mask.
        Defaults to flat and uint8.
    """
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # Validate mask format
    if mask.dtype != np.uint8:
        raise ValueError("Mask should be of type uint8")

    # Reduce dimensions if needed (some masks might be loaded as 3D)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Convert to one-hot encoding if requested
    if one_hot:
        if max_classes is None:
            raise ValueError(
                "max_classes must be provided when one_hot=True"
            )
        mask = to_categorical(mask, max_classes)

    # Convert to float if requested
    if to_float:
        mask = mask.astype(np.float32)

    return mask


def void_borders(
    mask: np.ndarray,
    border_width: int = 0,
    pad: int = 0,
):
    """
    Create a 2D mask with zeros in the class borders and external borders
    of the source mask.

    Args:
        mask (np.ndarray): Input mask.

        border_width (int, optional): Width of border to be voided.
        Defaults to 0.

        pad (int, optional): Amount of padding to be voided. Defaults to 0.

    Returns:
        np.ndarray: Mask with voided borders and padding.
    """
    assert border_width >= 0
    assert pad >= 0
    assert mask.ndim >= 2
    from scipy import ndimage

    void = np.ones(mask.shape[:2], dtype=np.uint8)
    if border_width > 0:
        # remove pixels near class borders
        void = np.zeros(mask.shape[:2], dtype=np.uint8)
        element = np.ones([border_width, border_width])
        if mask.ndim == 2:
            # case of flat mask
            classes = np.unique(mask)
            for cl in classes:
                m = np.where(mask == cl, 1, 0)
                void += ndimage.binary_erosion(
                    m, structure=element, border_value=0
                )
        else:
            # case of one-hot encoded mask
            classes = mask.shape[-1]
            for cl in range(classes):
                void += ndimage.binary_erosion(
                    mask[..., cl], structure=element, border_value=0
                )
        void[void > 0] = 1
    if pad > 0:
        # remove pixels along external borders
        void[:pad, :, ...] = 0
        void[-pad:, :, ...] = 0
        void[:, :pad, ...] = 0
        void[:, -pad:, ...] = 0

    return void


def _get_patch_coords(
    img_shape: tuple[int, ...], patch_size: int, patch_stride: int
):
    h, w = img_shape[:2]
    nh = math.ceil((h - patch_size) / patch_stride) + 1
    nw = math.ceil((w - patch_size) / patch_stride) + 1
    coords = []
    for i in range(nh):
        y = max(0, min(i * patch_stride, h - patch_size))
        for j in range(nw):
            x = max(0, min(j * patch_stride, w - patch_size))
            coords.append((y, x))
    return coords


def split_into_patches(
    img: np.ndarray,
    patch_size: int,
    patch_stride: int,
) -> list[np.ndarray]:
    """
    Splits image (>= 2 dimensions) into patches.

    Args:
        img (np.ndarray): source image
        patch_size (int): patch size in pixels
        patch_stride (int): patch stride in pixels

    Returns:
        List[np.ndarray]: list of extracted patches
    """
    assert img.ndim >= 2
    coords = _get_patch_coords(img.shape, patch_size, patch_stride)
    patches = []
    for coord in coords:
        y, x = coord
        patch = img[y : y + patch_size, x : x + patch_size, ...]
        patches.append(patch)
    return patches


def combine_from_patches(
    patches: Iterable[np.ndarray],
    patch_size: int,
    patch_stride: int,
    src_size: tuple[int, int],
) -> np.ndarray:
    """
    Combines patches back into image.

    Args:
        patches (Iterable[np.ndarray]): patches
        patch_size (int): patch size in pixels
        patch_stride (int): patch stride in pixels
        src_size (tuple[int, int]): target image shape

    Returns:
        np.ndarray: combined image
    """
    h, w = src_size[:2]
    target_shape = (h, w) + patches[0].shape[2:]
    img = np.zeros(target_shape, dtype=np.float32)
    density = np.zeros_like(img)
    coords = _get_patch_coords(img.shape, patch_size, patch_stride)
    for i, coord in enumerate(coords):
        y, x = coord
        y0, y1 = y, y + patch_size
        x0, x1 = x, x + patch_size
        img[y0:y1, x0:x1, ...] += patches[i]
        density[y0:y1, x0:x1, ...] += 1
    density[density == 0] = 1
    img /= density
    img = img.astype(patches[0].dtype)
    return img


def test_spit_combine_random(n_tests=100, eps=1e-3):
    from tqdm import tqdm

    for _ in tqdm(range(n_tests)):
        h = np.random.randint(100, 5000)
        w = np.random.randint(100, 5000)
        patch_s = np.random.randint(16, 1024)
        patch_s = min(h, w, patch_s)
        img = np.random.random((h, w))
        conv_offset = min(np.random.randint(50), patch_s // 4)
        overlay = np.random.randint(0, patch_s // 2)
        patches = split_into_patches(img, patch_s, conv_offset, overlay)
        img_reconstructed = combine_from_patches(
            patches, patch_s, conv_offset, overlay, img.shape
        )
        img_crop = img[conv_offset:-conv_offset, conv_offset:-conv_offset]
        img_reconstructed_crop = img_reconstructed[
            conv_offset:-conv_offset, conv_offset:-conv_offset
        ]
        assert np.sum(np.abs(img_crop - img_reconstructed_crop)) < eps
    logger.success("ok")


class BasicBatchCollector:
    """
    Basic class that collects patches into batches for non-PyTorch
    environments.

    This is a simple batch collector that doesn't support prefetching - it
    collects batches one at a time in a blocking manner. For PyTorch
    environments, it's recommended to use the SelfBalancingDataset's
    dataloaders instead, which leverages PyTorch's built-in prefetching
    capabilities.

    Args:
        patch_iter: Iterator yielding tuples of (image, mask) patches
        batch_s: Size of the batch
        img_postproc_func: Function to apply to the image patch before batching
        mask_postproc_func: Function to apply to the mask patch before batching

    Yields:
        Tuples of (image_batch, mask_batch) as numpy arrays
    """

    def __init__(
        self,
        patch_iter: Iterator[tuple[np.ndarray, np.ndarray]],
        batch_s: int,
    ) -> None:
        self.patch_iter = patch_iter
        self.batch_s = batch_s

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        x, y = [], []
        while True:
            img, mask = next(self.patch_iter)
            x.append(img)
            y.append(mask)
            if len(x) == self.batch_s:
                yield np.stack(x), np.stack(y)
                x.clear()
                y.clear()
