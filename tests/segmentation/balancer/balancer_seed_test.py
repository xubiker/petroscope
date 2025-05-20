import pytest

from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset
from petroscope.utils import logger
from tests.segmentation.balancer.utils import hash_patches

# Fixtures are imported automatically from conftest.py


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
    """
    Parameterized test to verify seed behavior with multiple seed
    combinations and sampling methods.
    """
    seed1, seed2 = seed_pair
    num_patches = balancer_params["num_patches"]
    same_seeds = seed1 == seed2

    logger.info(
        f"Testing {sampler_method} sampler with seeds {seed1} and {seed2}"
    )

    # Create balancers with the two seeds-
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
    patch_hashes1 = hash_patches(sampler1, num_patches)
    patch_hashes2 = hash_patches(sampler2, num_patches)

    # Count matches
    matches = sum(h1 == h2 for h1, h2 in zip(patch_hashes1, patch_hashes2))
    match_percentage = matches / num_patches * 100

    logger.info(
        f"{sampler_method.capitalize()} sampler seeds {seed1}/{seed2}: "
        f"{matches}/{num_patches} patches match ({match_percentage:.2f}%)"
    )

    # Check expectations based on whether seeds are the same
    if same_seeds:
        assert matches == num_patches, (
            "Patches should be identical with the same seed ",
            f"using {sampler_method} sampler",
        )
    else:
        assert match_percentage < 30.0, (
            "Too many matching patches between different seeds ",
            f"using {sampler_method} sampler",
        )
