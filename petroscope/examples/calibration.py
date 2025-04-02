"""
This script demonstrates how to use the ImageCalibrator class
from the petroscope.calibrate module to calibrate a batch of images.

You can also use the prepared script:
python -m petroscope.calibrate.run
"""

from pathlib import Path

from petroscope.calibrate.core import ImageCalibrator


def run_calibration():
    data_p = Path.cwd() / "data"
    calibrator = ImageCalibrator(
        reference_mirror_path=data_p / "mirror1200.jpg"
    )

    samples = ("train", "test")

    ds_path_in = Path("/mnt/c/dev/LumenStone/S1_v2/")
    ds_path_out = Path("/mnt/c/dev/LumenStone/S1_v2_calib/")

    for sample in samples:
        calibrator.calibrate_batch(
            ds_path_in / "imgs" / sample,
            ds_path_out / "imgs" / sample,
        )


if __name__ == "__main__":
    run_calibration()
