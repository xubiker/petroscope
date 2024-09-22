import argparse
from pathlib import Path

import petroscope.segmentation as segm
from petroscope.segmentation.utils.base import prepare_experiment


def run_inference(img_path: Path, out_dir: Path, device: str):

    classes = segm.classes.LumenStoneClasses.S1v1
    model = segm.models.ResUNetTorch.best(device)

    from petroscope.segmentation.utils.data import load_image
    from petroscope.segmentation.utils.vis import SegmVisualizer

    img = load_image(img_path)
    prediction = model.predict_image(img)
    prediction_colorized = SegmVisualizer.colorize_mask(
        prediction, classes.idx_to_colors, return_image=True
    )
    prediction_colorized.save(out_dir / f" {img_path.stem}_pred_colorized.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use for inference",
    )
    args = parser.parse_args()
    run_inference(
        img_path=Path("/mnt/c/dev/LumenStone/S1_v1_x05/imgs/test/01.jpg"),
        out_dir=prepare_experiment(Path("./out")),
        device=args.device,
    )
