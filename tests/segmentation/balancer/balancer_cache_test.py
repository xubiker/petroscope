import pytest
from pathlib import Path
import tempfile

from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset
from petroscope.utils import logger
from tests.segmentation.balancer.utils import hash_patches

# Fixtures are imported automatically from conftest.py


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
    """
    Test that confirms that the DsCacher doesn't affect reproducibility.
    """
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
        sampler_cached = (
            balancer_cached.sampler_balanced()
            if sampler_method == "balanced"
            else balancer_cached.sampler_random()
        )

        # Collect patches with caching
        patch_hashes_cached = hash_patches(sampler_cached, num_patches)

        # Second run without caching
        balancer_no_cache = ClassBalancedPatchDataset(
            img_mask_paths=image_mask_paths,
            patch_size=balancer_params["patch_size"],
            cache_dir=None,  # No caching
            seed=seed,
        )

        # Get the appropriate sampler
        sampler_no_cache = (
            balancer_no_cache.sampler_balanced()
            if sampler_method == "balanced"
            else balancer_no_cache.sampler_random()
        )

        # Collect patches without caching
        patch_hashes_no_cache = hash_patches(sampler_no_cache, num_patches)

        # Count matches between cached and non-cached runs
        matches = sum(
            h1 == h2
            for h1, h2 in zip(patch_hashes_cached, patch_hashes_no_cache)
        )
        match_percentage = (matches / num_patches) * 100

        logger.info(
            f"{sampler_method.capitalize()} sampler with and without cache: "
            f"{matches}/{num_patches} patches match ({match_percentage:.2f}%)"
        )

        # Patches should be identical regardless of caching
        error_msg = (
            f"Patches should be identical with same seed regardless of "
            f"caching in {sampler_method} sampler"
        )
        assert matches == num_patches, error_msg
