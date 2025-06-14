import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import petroscope.segmentation as segm
from petroscope.utils.base import prepare_experiment


def run_inference(img_path: Path, out_dir: Path, device: str):
    classes = segm.classes.LumenStoneClasses.S1_S2()
    model = segm.models.PSPNet.from_pretrained(
        Path.home()
        / "dev/petroscope/petroscope/segmentation/models/outputs/2025-06-12/20-36-39/models/best_test_miou_weights.pth",
        device,
    )

    from petroscope.segmentation.utils import load_image
    from petroscope.segmentation.vis import SegmVisualizer

    img = load_image(img_path)
    pred = model.predict_image(img, return_logits=False).astype(np.uint8)

    cv2.imwrite(
        out_dir / f"{img_path.stem}_pred.png",
        pred,
    )

    pred_colored = SegmVisualizer.colorize_mask(
        pred,
        classes.colors_map(squeezed=False),
    )
    cv2.imwrite(
        out_dir / f"{img_path.stem}_pred_colored.jpg",
        pred_colored,
    )

    v = SegmVisualizer.vis_prediction(
        img[:, :, ::-1],
        pred,
        classes,
        classes_squeezed=True,
    )
    cv2.imwrite(
        out_dir / f"{img_path.stem}_pred_composite.jpg",
        v,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use for inference",
    )
    args = parser.parse_args()

    input_dir = Path.home() / "dev/LumenStone/P1"

    out_dir = prepare_experiment(Path("./out"))

    img_paths = [
        i
        for i in input_dir.iterdir()
        if i.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for img_p in tqdm(img_paths, "Running inference on images"):
        run_inference(
            img_path=img_p,
            out_dir=out_dir,
            device=args.device,
        )
