import argparse
from pathlib import Path
from petroscope.panoramas import Stitcher

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# python panoramas_stitcher.py --input_dir=/Users/xubiker/dev/LumenStone/P1_v1/corrected/001 --device=cpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument(
        "--output_file",
        type=str,
        default="panorama.jpg",
    )
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    dir_path = Path(args.input_dir)
    assert dir_path.is_dir(), f"{dir_path} is not a directory"

    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stchr = Stitcher(device=args.device)

    img_paths = [
        img_p
        for img_p in dir_path.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    panorama = stchr.stitch(img_paths=img_paths, verbose=True)
    panorama.save(args.output_file, quality=95)
