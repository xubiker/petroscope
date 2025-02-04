"""
Calibration module for petroscope package.

This module provides an interface to perform calibration of images
using reference images of a mirror or an OLED screen.

"""

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from petroscope.utils import logger


class ImageCalibrator:
    def __init__(
        self,
        reference_mirror_path: str | Path = None,
        reference_screen_path: str | Path = None,
        correct_distortion: bool = False,
    ) -> None:
        # Ensure at least one reference image is provided
        if not reference_mirror_path and not reference_screen_path:
            raise ValueError(
                "At least one reference image path should be provided."
            )

        # Validate and convert paths
        self._ref_mirror_path = self._validate_path(reference_mirror_path)
        self._ref_screen_path = self._validate_path(reference_screen_path)

        # Ensure distortion correction is only used for screen calibration
        if correct_distortion and not self._ref_screen_path:
            raise ValueError(
                "Distortion correction is only available for "
                "calibration with screen image."
            )

        self.correct_distortion = correct_distortion

        # Load reference images
        self._lum_map_mirror = self._illumination_mirror(self._ref_mirror_path)
        self._lum_map_screen = self._illumination_screen(self._ref_screen_path)
        self._lum_map = self._illumination_final(
            self._lum_map_mirror, self._lum_map_screen
        )

        if correct_distortion:
            self._distortion = self._distortion_screen(self._ref_screen_path)

    @staticmethod
    def _validate_path(path: str | Path | None) -> Path | None:
        """Converts to Path and checks if the file exists."""
        if path is None:
            return None
        path = Path(path) if not isinstance(path, Path) else path
        if not path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        return path

    def _illumination_mirror(
        self, ref_img_path: Path | None
    ) -> np.ndarray | None:
        """Loads and preprocesses the reference mirror image."""
        if ref_img_path is None:
            return None
        img = (
            Image.open(ref_img_path)
            .convert("L")
            .filter(ImageFilter.GaussianBlur(radius=25))
        )
        mirror = np.array(img, dtype=np.float32) / 255
        illumination_mask = mirror + (1 - np.max(mirror))
        return illumination_mask

    def _illumination_screen(
        self, ref_img_path: Path | None
    ) -> np.ndarray | None:
        """Placeholder for screen calibration preparation."""
        if ref_img_path is None:
            return None
        raise NotImplementedError

    def _distortion_screen(self, ref_img_path: Path) -> np.ndarray:
        """Placeholder for screen distortion calibration."""
        raise NotImplementedError

    def _illumination_final(
        self,
        mirror_map: np.ndarray | None,
        screen_map: np.ndarray | None,
    ) -> np.ndarray:
        map_1ch = None
        if mirror_map is not None and screen_map is not None:
            map_1ch = (mirror_map + screen_map) / 2
        else:
            if mirror_map is not None:
                map_1ch = mirror_map
            if screen_map is not None:
                map_1ch = screen_map
        map_3ch = np.repeat(map_1ch[:, :, np.newaxis], 3, axis=2)
        return map_3ch

    def _correct_illumination_mirror(self, image: np.ndarray) -> np.ndarray:
        mirror = self._ref_img
        mask = mirror + (1 - np.max(mirror))
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        corrected_im = image / mask_3ch
        return corrected_im

    def _correct_illumination_screen(self, image: np.ndarray) -> np.ndarray:
        # Placeholder for screen-based illumination correction
        return image

    def _correct_distortion_screen(self, image: np.ndarray) -> np.ndarray:
        # Placeholder for screen-based distortion correction
        return image

    def calibrate(
        self, img_path: Path, out_path: Path, quiet: bool = False
    ) -> None:
        try:
            img = np.array(Image.open(img_path)).astype(np.float32) / 255

            img_corrected = img / self._lum_map

            if self.correct_distortion:
                # Placeholder for distortion correction
                img_corrected = img_corrected

            img_res = Image.fromarray((img_corrected * 255).astype(np.uint8))
            img_res.save(out_path)
            if not quiet:
                logger.info(f"Saved calibrated image to {out_path}")
        except Exception as e:
            logger.error(f"Error during calibration of {img_path}: {e}")

    def calibrate_batch(
        self,
        src: Iterable[Path] | Path,
        out: Path,
        quiet: bool = False,
    ) -> None:
        try:
            if isinstance(src, Path):
                if src.is_file():
                    src = [src]
                else:
                    if not src.is_dir():
                        raise ValueError(f"Input folder does not exist: {src}")
                    src = src.iterdir()
            src = [
                p
                for p in src
                if p.is_file() and p.suffix in (".jpg", ".png", ".bmp")
            ]
            if not src:
                logger.warning("No images found in the input folder.")
                return

            out.mkdir(parents=True, exist_ok=True)
            if not quiet:
                src = tqdm(src, desc="Calibrating images")

            for img_p in src:
                self.calibrate(img_p, out / img_p.name, quiet=True)
        except Exception as e:
            logger.error(f"Error during batch calibration: {e}")
