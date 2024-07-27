from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class GeoSegmModel(ABC):

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def load(self, saved_path: Path, **kwargs) -> None:
        pass

    @abstractmethod
    def train(
        self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs
    ) -> None:
        pass

    @abstractmethod
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        pass
