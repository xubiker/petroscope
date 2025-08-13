from glob import glob
from pathlib import Path

import pytest

from petroscope.utils import logger


@pytest.fixture(scope="module")
def image_mask_paths():
    """Fixture to provide image and mask paths for testing."""
    # The path to test data should be configured appropriately
    data_dir = Path.home() / "dev/LumenStone/S1v2/"
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
        "cache_dir": Path.home() / ".petroscope" / "balancer",
        "patch_size": 384,
        "num_patches": 20,  # Reduced for faster tests
    }


@pytest.fixture(scope="module")
def dataloader_params():
    """Fixture to provide common dataloader parameters."""
    return {
        "batch_size": 16,
        "n_workers": 4,
        "prefetch_factor": 4,
        "pin_memory": False,
    }
