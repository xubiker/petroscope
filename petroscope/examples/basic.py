from pathlib import Path

import numpy as np

from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.utils import load_image, load_mask


img_p = Path.home() / "dev/LumenStone/S1_v2/imgs/train/train_01.jpg"
mask_p = Path.home() / "dev/LumenStone/S1_v2/masks/train/train_01.png"

img = load_image(img_p)
mask1 = load_mask(mask_p)
mask2 = load_mask(mask_p, classes=LumenStoneClasses.S1())

print(img.shape, img.dtype, img.min(), img.max())
print(mask1.shape, mask1.dtype, mask1.min(), mask1.max(), np.unique(mask1))
print(mask2.shape, mask2.dtype, mask2.min(), mask2.max(), np.unique(mask2))
