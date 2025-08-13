from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator
import json

import yaml


@dataclass
class Class:
    """
    Data class representing a segmentation class.

    Attributes:
        label (str): The label of the class.
        color (str): The color of the class in hexadecimal RGB format
        (e.g. "#FF0000" for red).
        code (int): The code of the class.
        name (str, optional): The name of the class. Defaults to None.
    """

    label: str
    color: str
    code: int
    name: str = None

    def __post_init__(self):
        def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        # Ensure color is a hex string
        if not isinstance(self.color, str) or not self.color.startswith("#"):
            raise ValueError(
                "Color must be a hexadecimal string starting with '#'"
            )

        # Convert hex to RGB and BGR
        self.color_rgb = hex_to_rgb(self.color)
        self.color_bgr = self.color_rgb[::-1]
        self.color_hex = self.color  # Already in hex format

    def __repr__(self) -> str:
        return (
            f"[{self.code}, {self.label} ({self.name}), color: {self.color}]"
        )


class ClassSet:
    """
    Class representing a set of segmentation classes.
    """

    def __init__(self, classes: Iterable[Class]) -> None:
        self.classes = list(classes)
        # Precompute mappings
        self.code_to_idx = {cl.code: i for i, cl in enumerate(self.classes)}
        self.idx_to_code = {i: cl.code for i, cl in enumerate(self.classes)}
        self.code_to_class = {cl.code: cl for cl in self.classes}
        self.idx_to_color_rgb = {
            i: self._convert_color(cl.color)
            for i, cl in enumerate(self.classes)
        }
        self.code_to_color_rgb = {
            cl.code: self._convert_color(cl.color) for cl in self.classes
        }
        # Add BGR color mappings for OpenCV compatibility
        self.idx_to_color_bgr = {
            i: cl.color_bgr for i, cl in enumerate(self.classes)
        }
        self.code_to_color_bgr = {cl.code: cl.color_bgr for cl in self.classes}

    def __len__(self):
        return len(self.classes)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(cl.label for cl in self.classes)

    @staticmethod
    def _convert_color(color: str) -> tuple[int, int, int]:
        """Convert a hex color string to an RGB tuple."""
        hex_color = color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @property
    def idx_to_label(self) -> dict[int, str]:
        return {i: cl.label for i, cl in enumerate(self.classes)}

    @property
    def code_to_label(self) -> dict[int, str]:
        return {cl.code: cl.label for cl in self.classes}

    def colors_map(
        self, squeezed: bool, bgr=False
    ) -> dict[int, tuple[int, int, int]]:
        """
        Returns a mapping of class indices or codes to colors.

        Args:
            squeezed (bool): If True, returns mapping from indices to colors,
                            otherwise returns mapping from codes to colors.
            bgr (bool): If True, returns BGR colors for OpenCV,
                       otherwise returns RGB colors.

        Returns:
            dict[int, tuple[int, int, int]]: A mapping of indices or codes to colors.
        """
        if bgr:
            return (
                self.idx_to_color_bgr if squeezed else self.code_to_color_bgr
            )
        else:
            return (
                self.idx_to_color_rgb if squeezed else self.code_to_color_rgb
            )

    @property
    def labels_to_colors_plt(self) -> dict[str, tuple[float, float, float]]:
        def normalize_plt(
            r: int, g: int, b: int
        ) -> tuple[float, float, float]:
            return r / 255, g / 255, b / 255

        return {
            cl.label: normalize_plt(*self.code_to_color_rgb[cl.code])
            for cl in self.classes
        }

    def __iter__(self) -> Iterator[Class]:
        return iter(self.classes)

    def get_class_by_code(self, code: int) -> Class:
        """
        Get a Class object by its code.

        Args:
            code: The code of the class to retrieve

        Returns:
            Class object with the specified code

        Raises:
            KeyError: If no class with the specified code exists
        """
        if code not in self.code_to_class:
            raise KeyError(f"No class found with code {code}")
        return self.code_to_class[code]

    def to_json(self, filepath: str) -> None:
        """
        Save ClassSet to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "classes": [asdict(cls) for cls in self.classes],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "ClassSet":
        """
        Load ClassSet from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ClassSet instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        class_list = data.get("classes", [])
        classes = [Class(**class_data) for class_data in class_list]
        return cls(classes)


class LumenStoneClasses:
    _config = None
    _classes = None

    @classmethod
    def get_config(cls, yaml_path="lumenstone.yaml"):
        if cls._config is None:
            if type(yaml_path) is str:
                yaml_path = Path(__file__).parent / yaml_path
            with open(yaml_path, "r") as file:
                cls._config = yaml.safe_load(file)
        return cls._config

    @classmethod
    def all(cls) -> ClassSet:
        if cls._classes is None:
            cls._classes = [
                Class(**item) for item in cls.get_config()["classes"]
            ]
        return ClassSet(cls._classes)

    @classmethod
    def _classes_for_set(cls, name: str) -> list[Class]:
        v = cls.get_config()["sets"][name]
        return [cl for cl in cls.all().classes if cl.code in v]

    @classmethod
    def S1(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S1"))

    @classmethod
    def S2(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S2"))

    @classmethod
    def S3(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S3"))

    @classmethod
    def S1_S2(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S1_S2"))

    @classmethod
    def from_name(cls, name: str) -> ClassSet:
        func = getattr(cls, name)
        return func()

    @classmethod
    def from_ids(cls, ids: list[int]) -> ClassSet:
        """
        Create a ClassSet containing only the classes with the specified
        codes (ids).

        Args:
            ids (list[int]): List of class codes to include.

        Returns:
            ClassSet: A set containing only the specified classes.
        """
        selected = [cl for cl in cls.all().classes if cl.code in ids]
        return ClassSet(selected)
