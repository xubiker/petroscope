"""
pytest tests for verifying seed reproducibility in the SelfBalancingDataset.

This test suite checks that:
1. Using the same seed produces the same patches
2. Using different seeds produces different patches
3. Tests both sampler_balanced() and sampler_random() methods
4. Tests reproducibility with and without the DsCacher
"""

import pytest
import numpy as np
import hashlib
from pathlib import Path
from glob import glob
import tempfile

from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset
from petroscope.utils import logger


@pytest.fixture(scope="module")
def image_mask_paths():
    """Fixture to provide image and mask paths for testing."""
    # The path to test data should be configured appropriately
    data_dir = Path.home() / "dev/LumenStone/S1_v2/"
    img_pattern = "imgs/test/*.jpg"
    mask_pattern = "masks/test/*.png"

    # Skip if data directory doesn't exist
    if not data_dir.exists():
        pytest.skip("Test data directory not found")

    # Get image and mask paths
    image_paths = sorted([Path(p) for p in glob(str(data_dir / img_pattern))])
    mask_paths = sorted([Path(p) for p in glob(str(data_dir / mask_pattern))])

    if len(image_paths) == 0 or len(mask_paths) == 0:
        pytest.skip("No image or mask files found")

    if len(image_paths) != len(mask_paths):
        logger.warning(
            f"N of images ({len(image_paths)}) != N of masks ({len(mask_paths)})"
        )

    # Create and return image-mask pairs
    img_mask_pairs = list(zip(image_paths, mask_paths))
    logger.info(f"Found {len(img_mask_pairs)} image-mask pairs for testing")
    return img_mask_pairs


@pytest.fixture(scope="module")
def balancer_params():
    """Fixture to provide common balancer parameters."""
    return {
        "patch_size": 256,
        "num_patches": 20,  # Reduced for faster tests
        "cache_dir": Path.home() / ".petroscope" / "balancer",
    }


def get_patch_hash(img_patch, mask_patch):
    """Create a hash from image and mask patches to compare them."""
    # Combine image and mask data for hashing
    combined = np.concatenate([img_patch.flatten(), mask_patch.flatten()])
    # Create hash
    hash_object = hashlib.md5(combined.tobytes())
    return hash_object.hexdigest()


def collect_sample_patches(balancer, num_patches):
    """Collect a number of patches from the balancer and return their hashes."""
    patch_hashes = []

    # Get patches from the balancer
    for _ in range(num_patches):
        img_patch, mask_patch = next(balancer)
        patch_hash = get_patch_hash(img_patch, mask_patch)
        patch_hashes.append(patch_hash)

    return patch_hashes


@pytest.mark.parametrize(
    "seed_pair",
    [
        (42, 42),  # Same seeds - should match
        (105, 105),  # Same seeds - should match
        (42, 123),  # Different seeds - should differ
        (123, 456),  # Different seeds - should differ
    ],
)
@pytest.mark.parametrize(
    "sampler_method",
    [
        "balanced",  # Test sampler_balanced()
        "random",  # Test sampler_random()
    ],
)
def test_seed_behavior(
    image_mask_paths, balancer_params, seed_pair, sampler_method
):
    """Parameterized test to verify seed behavior with multiple seed combinations and sampling methods."""
    seed1, seed2 = seed_pair
    num_patches = balancer_params["num_patches"]
    same_seeds = seed1 == seed2

    logger.info(
        f"Testing {sampler_method} sampler with seeds {seed1} and {seed2}"
    )

    # Create balancers with the two seeds
    balancer1 = ClassBalancedPatchDataset(
        img_mask_paths=image_mask_paths,
        patch_size=balancer_params["patch_size"],
        cache_dir=balancer_params["cache_dir"],
        seed=seed1,
    )

    balancer2 = ClassBalancedPatchDataset(
        img_mask_paths=image_mask_paths,
        patch_size=balancer_params["patch_size"],
        cache_dir=balancer_params["cache_dir"],
        seed=seed2,
    )

    # Get the appropriate sampler method
    if sampler_method == "balanced":
        sampler1 = balancer1.sampler_balanced()
        sampler2 = balancer2.sampler_balanced()
    else:  # random
        sampler1 = balancer1.sampler_random()
        sampler2 = balancer2.sampler_random()

    # Get patches
    patch_hashes1 = collect_sample_patches(sampler1, num_patches)
    patch_hashes2 = collect_sample_patches(sampler2, num_patches)

    # Count matches
    matches = sum(h1 == h2 for h1, h2 in zip(patch_hashes1, patch_hashes2))
    match_percentage = matches / num_patches * 100

    logger.info(
        f"{sampler_method.capitalize()} sampler seeds {seed1}/{seed2}: "
        f"{matches}/{num_patches} patches match ({match_percentage:.2f}%)"
    )

    # Check expectations based on whether seeds are the same
    if same_seeds:
        assert (
            matches == num_patches
        ), f"Patches should be identical with the same seed using {sampler_method} sampler"
    else:
        assert (
            match_percentage < 30.0
        ), f"Too many matching patches between different seeds using {sampler_method} sampler"


@pytest.mark.parametrize(
    "sampler_method",
    [
        "balanced",  # Test sampler_balanced()
        "random",  # Test sampler_random()
    ],
)
def test_cacher_reproducibility(
    image_mask_paths, balancer_params, sampler_method
):
    """Test that using the DsCacher doesn't affect seed reproducibility."""
    seed = 42
    num_patches = balancer_params["num_patches"]

    # Create a temporary directory for caching
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        logger.info(f"Testing {sampler_method} sampler with caching")

        # First run with caching
        balancer_cached = ClassBalancedPatchDataset(
            img_mask_paths=image_mask_paths,
            patch_size=balancer_params["patch_size"],
            cache_dir=cache_dir,
            seed=seed,
        )

        # Get the appropriate sampler
        if sampler_method == "balanced":
            sampler_cached = balancer_cached.sampler_balanced()
        else:  # random
            sampler_cached = balancer_cached.sampler_random()

        # Collect patches with caching
        patch_hashes_cached = collect_sample_patches(
            sampler_cached, num_patches
        )

        # Second run without caching
        balancer_no_cache = ClassBalancedPatchDataset(
            img_mask_paths=image_mask_paths,
            patch_size=balancer_params["patch_size"],
            cache_dir=None,  # No caching
            seed=seed,
        )

        # Get the appropriate sampler
        if sampler_method == "balanced":
            sampler_no_cache = balancer_no_cache.sampler_balanced()
        else:  # random
            sampler_no_cache = balancer_no_cache.sampler_random()

        # Collect patches without caching
        patch_hashes_no_cache = collect_sample_patches(
            sampler_no_cache, num_patches
        )

        # Count matches between cached and non-cached runs
        matches = sum(
            h1 == h2
            for h1, h2 in zip(patch_hashes_cached, patch_hashes_no_cache)
        )
        match_percentage = matches / num_patches * 100

        logger.info(
            f"{sampler_method.capitalize()} sampler with and without cache: "
            f"{matches}/{num_patches} patches match ({match_percentage:.2f}%)"
        )

        # Patches should be identical regardless of caching
        assert (
            matches == num_patches
        ), f"Patches should be identical with same seed regardless of caching in {sampler_method} sampler"
