from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np
import math


@dataclass
class Class:
    label: str
    color: str | tuple[int, int, int]
    code: int
    name: str = None


class ClassAssociation:
    def __init__(self, classes: Iterable[Class]) -> None:
        self.classes = list(classes)

    def __len__(self):
        return len(self.classes)

    @property
    def labels(self) -> tuple[int, ...]:
        return [cl.label for cl in self.classes]

    @property
    def squeeze_map(self) -> dict[int, int]:
        return {m.code: i for i, m in enumerate(self.classes)}

    @property
    def codes_to_colors(self) -> dict[int, tuple[int, int, int]]:
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def convert_color(color):
            if isinstance(color, str):
                return hex_to_rgb(color)
            return color

        return {i: convert_color(m.color) for i, m in enumerate(self.classes)}

    @property
    def codes_to_labels(self) -> dict[int, str]:
        return {i: m.label for i, m in enumerate(self.classes)}

    @property
    def labels_to_colors_plt(self) -> dict[str, tuple[int, int, int]]:
        def normalize_plt(
            r: int, g: int, b: int
        ) -> tuple[float, float, float]:
            return r / 255, g / 255, b / 255

        d = self.codes_to_colors
        return {
            label: normalize_plt(*d[code])
            for code, label in self.codes_to_labels.items()
        }


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
        A numpy array of shape (H, W, n_classes) where each value is either 0 or 1,
        indicating the presence or absence of a class at that location.
    """
    assert x.ndim == 2
    input_shape = x.shape
    x = x.reshape(-1)
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, n_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def _squeeze_mask(
    mask: np.ndarray, squeeze: dict[int, int], void_val: int | None = 255
) -> np.ndarray:
    """
    Squeezes the mask by replacing certain values with others. Supports void
    values (can be used in masks near class borders).

    Args:
        mask (np.ndarray): The input mask to be squeezed.
        squeeze (dict[int, int]): A dictionary mapping original values to new
        values.
        void_val (bool | None, optional): Void value that will not be replaced.
        Defaults to 255. If None, void values will not be processed.

    Returns:
        np.ndarray: The squeezed mask.
    """
    new_mask = np.zeros_like(mask)
    for i, j in squeeze.items():
        new_mask[mask == i] = j
    if void_val is not None:
        new_mask[mask == void_val] = void_val
    return new_mask


def _preprocess_mask(
    mask: np.ndarray,
    squeeze: dict[int, int] | None,
    one_hot=True,
):
    # reduce dimensions
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # squeeze mask
    new_mask = _squeeze_mask(mask, squeeze) if squeeze is not None else mask
    # convert to one-hot
    return (
        to_categorical(new_mask, len(squeeze)).astype(np.float32)
        if one_hot
        else new_mask
    )


def load_image(path: Path, normalize=True):
    from PIL import Image

    img = np.array(Image.open(path)).astype(np.uint8)
    if normalize:
        img = img.astype(np.float32) / 255
    return img


def load_mask(
    path: Path,
    classes: ClassAssociation,
    one_hot=True,
):
    from PIL import Image

    return _preprocess_mask(
        np.array(Image.open(path)),
        classes.squeeze_map,
        one_hot=one_hot,
    )


def void_borders(
    mask: np.ndarray,
    border_width: int = 0,
    pad: int = 0,
):
    """
    Make a 2D mask with zeros in the class borders and external borders of the source mask.
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
    img_shape: tuple[int, ...], patch_size: int, conv_offset: int, overlay: int
):
    h, w = img_shape[:2]
    pps = patch_size - 2 * conv_offset
    s = pps - overlay
    nh = math.ceil((h - 2 * conv_offset) / s)
    nw = math.ceil((w - 2 * conv_offset) / s)
    coords = []
    for i in range(nh):
        y = min(i * s, h - patch_size)
        for j in range(nw):
            x = min(j * s, w - patch_size)
            coords.append((y, x))
    return coords


def split_into_patches(
    img: np.ndarray,
    patch_size: int,
    conv_offset: int,
    overlay: int | float,
) -> list[np.ndarray]:
    """
    Splits image (>= 2 dimensions) into patches.

    Args:
        img (np.ndarray): source image
        patch_size (int): patch size in pixels
        conv_offset (int): conv offset in pixels
        overlay (int | float): either float in [0, 1]
        (fraction of patch size) or int in pixels

    Returns:
        List[np.ndarray]: list of extracted patches
    """
    assert img.ndim >= 2
    if isinstance(overlay, float):
        overlay = int(patch_size * overlay)
    coords = _get_patch_coords(img.shape, patch_size, conv_offset, overlay)
    patches = []
    for coord in coords:
        y, x = coord
        patch = img[y : y + patch_size, x : x + patch_size, ...]
        patches.append(patch)
    return patches


def combine_from_patches(
    patches: Iterable[np.ndarray],
    patch_s: int,
    conv_offset: int,
    overlay: int | float,
    src_size: tuple[int, int],
    border_fill_val=0,
) -> np.ndarray:
    """
    Combines patches back into image.

    Args:
        patches (Iterable[np.ndarray]): patches
        patch_s (int): patch size in pixels
        conv_offset (int): conv offset in pixels
        overlay (Union[int, float]): either float in [0, 1]
        (fraction of patch size) or int in pixels
        src_size (Tuple[int, int]): target image shape
        border_fill_val (int, optional): value to fill the
        conv offset border. Defaults to 0.

    Returns:
        np.ndarray: combined image
    """
    if isinstance(overlay, float):
        overlay = int(patches[0].shape[0] * overlay)
    h, w = src_size[:2]
    target_shape = (h, w) + patches[0].shape[2:]
    img = np.zeros(target_shape, dtype=np.float32) + border_fill_val
    density = np.zeros_like(img)
    coords = _get_patch_coords(img.shape, patch_s, conv_offset, overlay)
    for i, coord in enumerate(coords):
        y, x = coord
        y0, y1 = y + conv_offset, y + patch_s - conv_offset
        x0, x1 = x + conv_offset, x + patch_s - conv_offset
        img[y0:y1, x0:x1, ...] += patches[i][
            conv_offset : patch_s - conv_offset,
            conv_offset : patch_s - conv_offset,
            ...,
        ]
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
    print("ok")


class BatchPacker:

    def __init__(
        self,
        patch_iter: Iterator[tuple[np.ndarray, np.ndarray]],
        batch_s: int,
        squeeze_map: dict[int, int],
        normalize_img: bool,
        one_hot: bool,
    ) -> None:
        self.patch_iter = patch_iter
        self.batch_s = batch_s
        self.squeeze_map = squeeze_map
        self.normalize_img = normalize_img
        self.one_hot = one_hot

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        x, y = [], []
        while True:
            img, mask = next(self.patch_iter)
            if self.normalize_img:
                img = img.astype(np.float32) / 255
            mask = _preprocess_mask(
                mask, self.squeeze_map, one_hot=self.one_hot
            )
            x.append(img)
            y.append(mask)
            if len(x) == self.batch_s:
                yield np.stack(x), np.stack(y)
                x.clear()
                y.clear()
