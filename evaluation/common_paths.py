#!/usr/bin/env python3
"""Common path and dataset discovery helpers for AVBench evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def project_root() -> Path:
    """Return AVBench project root inferred from this file location."""
    return Path(__file__).resolve().parents[1]


def default_video_root() -> Path:
    """Default root directory that stores evaluation videos."""
    return project_root() / "video_data"


def default_dataset_root() -> Path:
    """Default root directory that stores text/prompt datasets."""
    return project_root() / "dataset"


def default_results_root() -> Path:
    """Default output directory for evaluation reports."""
    return project_root() / "results"


def _has_video_files(dir_path: Path) -> bool:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    return any(p.is_file() and p.suffix.lower() in exts for p in dir_path.iterdir())


def discover_video_dirs(video_root: Path) -> List[Tuple[str, str]]:
    """
    Discover evaluation datasets under video_root.

    Rule:
    - If video files exist directly under video_root, return one dataset named "video_data".
    - Otherwise, return each immediate subdirectory that contains at least one video file.
    """
    video_root = Path(video_root)
    if not video_root.exists():
        return []

    if _has_video_files(video_root):
        return [("video_data", str(video_root))]

    pairs: List[Tuple[str, str]] = []
    for sub in sorted(video_root.iterdir()):
        if sub.is_dir() and _has_video_files(sub):
            pairs.append((sub.name, str(sub)))
    return pairs
