import hashlib
from typing import Iterator

import numpy as np
from tqdm import tqdm


def get_patch_hash(img_patch, mask_patch):
    """Create a hash from image and mask patches to compare them."""
    # Combine image and mask data for hashing
    combined = np.concatenate([img_patch.flatten(), mask_patch.flatten()])
    # Create hash
    hash_object = hashlib.md5(combined.tobytes())
    return hash_object.hexdigest()


def hash_patches(sampler, num_patches):
    """Collect a number of patches from the sampler and return their hashes."""
    patch_hashes = []

    # Get patches from the sampler
    for _ in range(num_patches):
        img_patch, mask_patch = next(sampler)
        patch_hash = get_patch_hash(img_patch, mask_patch)
        patch_hashes.append(patch_hash)

    return patch_hashes


def check_balancing_quality(
    masks_iterator: Iterator[np.ndarray], num_patches: int
) -> float:
    """
    Calculate the balancing quality of the masks.
    The balancing quality is defined as 1 - (max - min) / sum,
    where max and min are the maximum and minimum counts of
    unique values in the masks, and sum is the total count of
    all unique values.
    """

    d = dict()
    for _ in tqdm(range(num_patches), "Calculating balancing quality"):
        mask = next(masks_iterator)
        mask_values, counts = np.unique(mask, return_counts=True)
        for v, c in zip(mask_values, counts):
            if v == 255:
                continue
            d[v] = d.get(v, 0) + c

    quality = 1 - (max(d.values()) - min(d.values())) / sum(d.values())
    return quality
