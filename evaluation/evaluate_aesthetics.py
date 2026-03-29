#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate frame-level visual aesthetics for AVBench videos with aesthetic-predictor-v2-5."""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from PIL import Image
import torch
import numpy as np

try:
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
except ImportError as exc:
    ap_root = os.environ.get("AESTHETIC_PREDICTOR_ROOT")
    if ap_root:
        sys.path.insert(0, ap_root)
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    else:
        raise ImportError(
            "Cannot import aesthetic_predictor_v2_5. "
            "Set AESTHETIC_PREDICTOR_ROOT or install the package from "
            "https://github.com/discus0434/aesthetic-predictor-v2-5"
        ) from exc
from common_paths import default_results_root, default_video_root, discover_video_dirs


def find_video_files(video_dir: str) -> List[str]:
    exts = [".mp4", ".mov", ".avi", ".mkv"]
    p = Path(video_dir)
    files = [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]
    return sorted(files)


def extract_video_id(video_path: str) -> str:
    return Path(video_path).stem


def extract_frames(video_path: str, out_dir: str, fps: float = 1.0) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        "-vsync",
        "0",
        "-y",
        out_pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted([str(p) for p in Path(out_dir).glob("frame_*.jpg")])
    return frames


def score_frames(frames: List[str], model, preprocessor, device: torch.device, batch_size: int = 8):
    scores = []
    for i in range(0, len(frames), batch_size):
        batch_paths = frames[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = preprocessor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values
        try:
            pixel_values = pixel_values.to(device)
        except Exception:
            pixel_values = pixel_values

        with torch.inference_mode():
            out = model(pixel_values)
            # output shape (batch, 1)
            batch_scores = out.logits.squeeze().float().cpu().numpy()
            if batch_scores.ndim == 0:
                batch_scores = np.array([float(batch_scores)])
            scores.extend(batch_scores.tolist())

    return scores


def evaluate_videos(video_dir: str, output_path: str, fps: float = 1.0, batch_size: int = 8):
    video_files = find_video_files(video_dir)
    if not video_files:
        print(f"No videos found in {video_dir} ")
        return

    # load model
    print("Loading aesthetic-predictor-v2-5 model...")
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
    except Exception:
        pass

    tmp_root = tempfile.mkdtemp(prefix="va_aesthetic_")
    print(f"Temporary frame directory: {tmp_root}")

    rows = []
    try:
        for vid in video_files:
            vid_id = extract_video_id(vid)
            video_tmp = os.path.join(tmp_root, vid_id)
            os.makedirs(video_tmp, exist_ok=True)
            print(f"Processing: {vid} -> extracting frames (fps={fps})")
            try:
                frames = extract_frames(vid, video_tmp, fps=fps)
            except subprocess.CalledProcessError:
                print(f"ffmpeg processing failed: {vid}")
                continue

            if not frames:
                print(f"No frames extracted: {vid}")
                continue

            scores = score_frames(frames, model, preprocessor, device, batch_size=batch_size)
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            rows.append({
                "video_id": vid_id,
                "video_path": vid,
                "n_frames": len(frames),
                "mean_score": mean,
                "std_score": std,
            })

        # Write CSV output.
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "video_path", "n_frames", "mean_score", "std_score"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Evaluation completed. Results saved to {output_path}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    p.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    p.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (frames per second).")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    fps = args.fps
    batch_size = args.batch_size
    results_dir = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"No video folders found under: {args.video_root}")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {label}  ({video_dir})")
        print(f"{'#'*60}")
        if not os.path.exists(video_dir):
            print("Skipping: directory does not exist")
            continue
        output_path = os.path.join(results_dir, f"aesthetics_{label}.csv")
        rows = evaluate_videos(video_dir, output_path, fps=fps, batch_size=batch_size)
        if rows:
            scores = [r["mean_score"] for r in rows]
            summary_rows.append({
                "dataset":    label,
                "n_videos":   len(rows),
                "mean_score": float(np.mean(scores)),
                "std_score":  float(np.std(scores)),
            })

    summary_path = os.path.join(results_dir, "aesthetics_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "n_videos", "mean_score", "std_score"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"\n{'='*60}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*60}")
    for r in summary_rows:
        print(f"  {r['dataset']:20s}: n={r['n_videos']:4d}  mean={r['mean_score']:.4f}  std={r['std_score']:.4f}")


if __name__ == "__main__":
    main()
