from pathlib import Path

import yaml

from petroscope.segmentation.utils.data import Class, ClassSet


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
    def S1v1(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S1v1"))

    @classmethod
    def S2v1(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S2v1"))

    @classmethod
    def S3v1(cls) -> ClassSet:
        return ClassSet(cls._classes_for_set("S3v1"))

    @classmethod
    def from_name(cls, name: str) -> ClassSet:
        func = getattr(cls, name)
        return func()
