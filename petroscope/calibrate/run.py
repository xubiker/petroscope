import argparse
from pathlib import Path

from petroscope.calibrate import ImageCalibrator
from petroscope.utils import logger


def run_calibration():
    parser = argparse.ArgumentParser(description="Batch image calibration.")
    parser.add_argument(
        "-m",
        "--mirror",
        type=Path,
        help="Path to the reference image of mirror.",
    )
    parser.add_argument(
        "-s",
        "--screen",
        type=Path,
        help="Path to the reference image of screen.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the output folder.",
    )
    parser.add_argument(
        "-d",
        "--distortion",  # FIX: Removed extra comma
        action="store_true",
        help="Enable distortion correction "
        "(only if screen reference image is passed).",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting calibration...")
        calibrator = ImageCalibrator(
            reference_mirror_path=args.mirror,
            reference_screen_path=args.screen,
            correct_distortion=args.distortion,
        )
        calibrator.calibrate_batch(args.input, args.output)
        logger.success("Calibration completed.")
    except Exception as e:
        logger.error(f"Error during calibration: {e}")


if __name__ == "__main__":
    run_calibration()
