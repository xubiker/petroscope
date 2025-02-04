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

    calibrator.calibrate_batch(
        data_p / "calibration_src",
        data_p / "calibration_out",
    )


if __name__ == "__main__":
    run_calibration()
