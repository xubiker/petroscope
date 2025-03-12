import argparse
from pathlib import Path

import petroscope.segmentation as segm
from petroscope.utils.base import prepare_experiment


def run_inference(img_path: Path, out_dir: Path, device: str):
    classes = segm.LumenStoneClasses.S1v1()
    model = segm.models.ResUNetTorch.trained("s1_x05", device)

    from petroscope.segmentation.utils import load_image
    from petroscope.segmentation.vis import SegmVisualizer

    img = load_image(img_path)
    prediction = model.predict_image(img, retutn_logits=False)
    v = SegmVisualizer.vis_prediction(
        img,
        prediction,
        classes,
        classes_squeezed=True,
    )
    v.save(out_dir / f" {img_path.stem}_pred.jpg", quality=95)


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
        img_path=Path(
            "/Users/xubiker/dev/LumenStone/S1_v1.5/imgs/test/test_01.jpg"
        ),
        out_dir=prepare_experiment(Path("./out")),
        device=args.device,
    )
