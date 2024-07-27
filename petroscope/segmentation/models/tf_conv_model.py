from pathlib import Path
from typing import Iterable, Iterator
from numpy import ndarray
from petroscope.segmentation.model import GeoSegmModel


class TFConvModel(GeoSegmModel):
    def initialize(self) -> None:
        raise NotImplementedError

    def load(self, saved_path: Path, **kwargs) -> None:
        raise NotImplementedError

    def train(
        self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs
    ) -> None:
        raise NotImplementedError

    def predict_image(self, image: ndarray) -> ndarray:
        raise NotImplementedError
