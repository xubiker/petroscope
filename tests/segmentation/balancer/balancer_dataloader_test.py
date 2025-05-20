import pytest
from petroscope.segmentation.balancer.balancer import ClassBalancedPatchDataset
from petroscope.utils import logger
from tests.segmentation.balancer.utils import check_balancing_quality

# Fixtures are imported automatically from conftest.py


@pytest.mark.parametrize("num_patches", [100, 1000])
def test_balancing_quality_comparison(
    image_mask_paths, balancer_params, dataloader_params, num_patches
):
    """
    Compare balancing quality between multiprocess dataloader_balanced()
    and sampler_balanced().
    """
    import torch

    seed = 42

    def sampler_iterator():
        logger.info("Creating balancer dataset instance")
        balancer = ClassBalancedPatchDataset(
            img_mask_paths=image_mask_paths,
            patch_size=balancer_params["patch_size"],
            cache_dir=balancer_params["cache_dir"],
            seed=seed,
        )
        sampler = balancer.sampler_balanced()
        while True:
            _, mask = next(sampler)
            yield mask

    def dataloader_iterator():
        logger.info("Creating balancer dataset instance")
        balancer = ClassBalancedPatchDataset(
            img_mask_paths=image_mask_paths,
            patch_size=balancer_params["patch_size"],
            cache_dir=balancer_params["cache_dir"],
            seed=seed,
        )

        dataloader = balancer.dataloader_balanced(
            batch_size=dataloader_params["batch_size"],
            num_workers=dataloader_params["n_workers"],
            pin_memory=dataloader_params["pin_memory"],
            prefetch_factor=dataloader_params["prefetch_factor"],
        )

        for _, mask_batch in dataloader:
            if isinstance(mask_batch, torch.Tensor):
                mask_batch = mask_batch.numpy()
            yield mask_batch  # no need to split into patches

    bs = dataloader_params["batch_size"]

    num_patches = num_patches // bs * bs  # Adjust for batch size

    it1 = sampler_iterator()

    it2 = dataloader_iterator()

    quality_1 = check_balancing_quality(it1, num_patches)
    logger.info(f"simple sampler balancing quality: {quality_1:.4f}")
    quality_2 = check_balancing_quality(it2, num_patches // bs)

    logger.info(f"dataloader balancing quality: {quality_2:.4f}")

    # We don't require the qualities to be exactly the same as they use
    # different mechanisms, but they should be reasonably close if both
    # implementations are working correctly
    assert abs(quality_1 - quality_2) < 0.15, (
        "Balancing quality difference between sampler and dataloader "
        "is too high"
    )
