from pathlib import Path
from typing import Iterable


def _units_formatter(base: int, labels: Iterable[str]):
    def formatter(size: int) -> str:
        lbl_idx = 0
        while size > base and lbl_idx < len(labels) - 1:
            size = size / base
            lbl_idx += 1
        return f"{size:.2f}{labels[lbl_idx]}"

    return formatter


formatter_bytes = _units_formatter(
    base=1024, labels=("B", "KB", "MB", "GB", "TB")
)
formatter_si = _units_formatter(
    base=1000, labels=("", "K", "M", "G", "T", "P", "E", "Z")
)
formatter_count = _units_formatter(base=1000, labels=("", "K", "M", "B", "T"))


def prepare_experiment(out_path: Path) -> Path:
    out_path.mkdir(parents=True, exist_ok=True)
    dirs = list(out_path.iterdir())
    dirs = [d for d in dirs if d.name.startswith("exp_")]
    experiment_id = (
        max(int(d.name.split("_")[1]) for d in dirs) + 1 if dirs else 1
    )
    exp_path = out_path / f"exp_{experiment_id}"
    exp_path.mkdir()
    return exp_path
