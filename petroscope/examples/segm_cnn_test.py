import argparse
from pathlib import Path

import petroscope.segmentation as segm
from petroscope.segmentation.utils.base import prepare_experiment


def get_test_img_mask_pairs(ds_dir: Path):
    """
    Get paths to test images and corresponding masks from dataset directory.
    """
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def run_test(
    ds_dir: Path,
    out_dir: Path,
    device: str,
    void_pad=4,
    void_border_width=2,
    vis_segmentation=True,
    vis_plots=False,
):
    """
    Runs model on test images from dataset directory and
    saves results to output directory.
    """
    classes = segm.LumenStoneClasses.S1v1()
    # create the model (PSPNetTorch or ResUnetTorch) and load weights
    model = segm.models.PSPNetTorch.trained("s1_resnet18_x05_calib", device)
    # model = segm.models.ResUNetTorch.trained("s1_x05", device)

    tester = segm.SegmDetailedTester(
        out_dir=out_dir,
        classes=classes,
        void_pad=void_pad,
        void_border_width=void_border_width,
        vis_segmentation=vis_segmentation,
        vis_plots=vis_plots,
    )
    res, res_void = tester.test_on_set(
        get_test_img_mask_pairs(ds_dir),
        predict_func=model.predict_image,
        description="images",
    )
    print("results without void borders:\n", res)
    print("results with void borders:\n", res_void)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use for inference",
    )
    args = parser.parse_args()
    run_test(
        ds_dir=Path("/mnt/c/dev/LumenStone/S1_v1_x05_calib"),
        out_dir=prepare_experiment(Path("./out")),
        device=args.device,
    )
