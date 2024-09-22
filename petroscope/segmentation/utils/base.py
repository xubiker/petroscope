from pathlib import Path
from typing import Iterable
from datetime import datetime


class UnitsFormatter:

    def _units_formatter(base: int, labels: Iterable[str]):
        def formatter(size: int) -> str:
            lbl_idx = 0
            while size > base and lbl_idx < len(labels) - 1:
                size = size / base
                lbl_idx += 1
            return f"{size:.2f}{labels[lbl_idx]}"

        return formatter

    @staticmethod
    def bytes(size: int) -> str:
        return UnitsFormatter._units_formatter(
            base=1024, labels=("B", "KB", "MB", "GB", "TB")
        )(size)

    @staticmethod
    def si(size: int) -> str:
        return UnitsFormatter._units_formatter(
            base=1000, labels=("", "K", "M", "G", "T", "P", "E", "Z")
        )(size)

    @staticmethod
    def count(size: int) -> str:
        return UnitsFormatter._units_formatter(
            base=1000, labels=("", "K", "M", "B", "T")
        )(size)


def prepare_experiment(out_path: Path) -> Path:
    out_path.mkdir(parents=True, exist_ok=True)
    dirs = list(out_path.iterdir())
    dirs = [d for d in dirs if d.name.startswith("exp-")]
    experiment_id = (
        max(int(d.name.split("_")[0].split("-")[1]) for d in dirs) + 1
        if dirs
        else 1
    )
    t = datetime.now()
    exp_path = out_path / f"exp-{experiment_id:03}_{t:%Y-%m-%d_%H-%M}"
    exp_path.mkdir()
    return exp_path
