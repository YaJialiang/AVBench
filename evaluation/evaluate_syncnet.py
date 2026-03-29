#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate lip-sync quality of AVBench videos with SyncNet (LatentSync)."""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import torch
import cv2

from common_paths import default_results_root, default_video_root, discover_video_dirs


def find_video_files(video_dir: str) -> List[str]:
    """Find all video files under a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    return sorted([str(f) for f in video_files])


def extract_video_id(video_path: str) -> str:
    """Extract stable video id from a generated filename."""
    filename = Path(video_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        video_id = f"{parts[0]}_{parts[1]}"
        return video_id
    return filename


def compute_sync_score(confidence: float, offset_frames: int) -> float:
    """
        Compute a combined lip-sync score in [0, 100].

        Intuition:
        - high confidence + small offset -> high score
        - high confidence + large offset -> low score (offset penalty is stronger)
        - low confidence + any offset -> medium/low score

        Formula:
            conf_score   = sigmoid((conf - 5.0) * 0.5)    # confidence normalization
            offset_decay = exp(-|offset_frames| / 3.0)    # offset decay, ~half-life at 3 frames
      sync_score   = conf_score * offset_decay * 100
    """
    import math
    conf_score   = 1.0 / (1.0 + math.exp(-(confidence - 5.0) * 0.5))
    offset_decay = math.exp(-abs(offset_frames) / 3.0)
    return round(conf_score * offset_decay * 100.0, 2)


def evaluate_lip_sync(video_path: str,
                     syncnet_eval: Any,
                     temp_dir: str,
                     batch_size: int = 20,
                     vshift: int = 15) -> Dict:
    """Evaluate lip-sync quality for one video."""
    try:
        # Source FPS is metadata only. SyncNet internally resamples to 25 FPS.
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if src_fps <= 0:
            src_fps = 25.0

        # Compatibility:
        # - Upstream LatentSync usually returns 3 values.
        # - Some local forks return 4 values with extra fps metadata.
        eval_ret = syncnet_eval.evaluate(
            video_path,
            temp_dir=temp_dir,
            batch_size=batch_size,
            vshift=vshift,
        )

        if isinstance(eval_ret, (tuple, list)) and len(eval_ret) >= 4:
            offset, min_dist, conf, ret_fps = eval_ret[:4]
            if ret_fps is not None and float(ret_fps) > 0:
                src_fps = float(ret_fps)
        else:
            offset, min_dist, conf = eval_ret[:3]

        # Cast tensors/scalars to Python native types for serialization.
        offset   = int(offset)
        conf     = float(conf)
        min_dist = float(min_dist)

        # Keep offset seconds based on 25 FPS for cross-run comparability.
        EVAL_FPS  = 25.0
        offset_sec = offset / EVAL_FPS
        
        # Bucket confidence into readable quality bands.
        if conf >= 7.0:
            sync_quality = 'Excellent'
        elif conf >= 5.0:
            sync_quality = 'Good'
        elif conf >= 3.0:
            sync_quality = 'Fair'
        else:
            sync_quality = 'Poor'

        # Final score combines confidence and temporal offset.
        sync_score = compute_sync_score(conf, offset)

        return {
            'offset_frames': offset,
            'offset_sec': offset_sec,
            'src_fps': round(src_fps, 3),
            'confidence': conf,
            'min_dist': min_dist,
            'sync_quality': sync_quality,
            'sync_score': sync_score,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'offset_frames': None,
            'offset_sec': None,
            'confidence': None,
            'min_dist': None,
            'sync_quality': None,
            'success': False,
            'error': str(e)
        }


def evaluate_videos(video_dir: str, 
                    output_path: str,
                    latentsync_root: str,
                    syncnet_ckpt: str,
                    device: str = 'cuda',
                    batch_size: int = 20,
                    vshift: int = 15):
    """Evaluate lip-sync quality for all videos in a directory."""
    print("=" * 80)
    print("SyncNet lip-sync evaluation")
    print("=" * 80)
    
    # Discover input videos.
    video_files = find_video_files(video_dir)
    print(f"\nFound {len(video_files)} video files")
    
    if not video_files:
        print(f"Error: no videos found in {video_dir}")
        return
    
    # Initialize SyncNet from a configurable LatentSync repository path.
    print(f"\nInitializing SyncNet model...")
    sys.path.insert(0, latentsync_root)
    from eval.syncnet.syncnet_eval import SyncNetEval
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    syncnet_eval = SyncNetEval(device=device)
    
    # Load pretrained weights when available.
    if os.path.exists(syncnet_ckpt):
        checkpoint = torch.load(syncnet_ckpt, map_location=device)
        
        # Support multiple checkpoint formats.
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        syncnet_eval.__S__.load_state_dict(state_dict, strict=False)
        print(f"SyncNet model loaded: {syncnet_ckpt}")
    else:
        print(f"Warning: SyncNet checkpoint not found: {syncnet_ckpt}")
        print("Using uninitialized weights; results may be inaccurate.")
    
    syncnet_eval.__S__.eval()
    
    # Create temporary workspace.
    temp_base_dir = "/tmp/syncnet_eval"
    os.makedirs(temp_base_dir, exist_ok=True)
    
    # Evaluation result
    results = []
    failed_videos = []
    
    print(f"\nStarting lip-sync evaluation...")
    print(f"Parameters: batch_size={batch_size}, vshift={vshift} frames")
    
    for video_path in tqdm(video_files, desc="Evaluating"):
        video_id = extract_video_id(video_path)
        video_name = Path(video_path).name
        
        # Use per-video temp directories to avoid file collisions.
        temp_dir = os.path.join(temp_base_dir, video_id)
        
        try:
            # Run lip-sync evaluation for this sample.
            result = evaluate_lip_sync(
                video_path, 
                syncnet_eval, 
                temp_dir,
                batch_size=batch_size,
                vshift=vshift
            )
            
            if result['success']:
                results.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'src_fps': result['src_fps'],        # source metadata
                    'offset_frames': result['offset_frames'],
                    'offset_sec': result['offset_sec'],
                    'confidence': result['confidence'],
                    'min_dist': result['min_dist'],
                    'sync_quality': result['sync_quality'],
                    'sync_score': result['sync_score'],
                })
            else:
                failed_videos.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'error': result['error']
                })
                print(f"\nError: failed to evaluate {video_name}: {result['error']}")
            
        except Exception as e:
            print(f"\nError: failed to evaluate {video_name}: {str(e)}")
            failed_videos.append({
                'video_id': video_id,
                'video_name': video_name,
                'error': str(e)
            })
    
    # Clean temporary workspace.
    import shutil
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir, ignore_errors=True)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('video_id')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nEvaluation completed. Results saved to: {output_path}")
        return df
    else:
        print("\nNo videos were evaluated successfully")
        return pd.DataFrame()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lip-sync quality with SyncNet.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--latentsync-root", default=os.environ.get("LATENTSYNC_ROOT", "/public/yangjl/LatentSync"), help="Path to LatentSync repository.")
    parser.add_argument("--syncnet-ckpt", default=None, help="Path to syncnet_v2.model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    latentsync_root = args.latentsync_root
    syncnet_ckpt = args.syncnet_ckpt or os.path.join(latentsync_root, "checkpoints", "auxiliary", "syncnet_v2.model")
    device = args.device
    batch_size = args.batch_size
    vshift = args.vshift
    results_dir  = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"No evaluable video directories found under: {args.video_root}")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*80}")
        print(f"# Dataset: {label}  ({video_dir})")
        print(f"{'#'*80}")
        if not os.path.exists(video_dir):
            print("Skipping: directory does not exist")
            continue
        output_path = os.path.join(results_dir, f"syncnet_{label}.csv")
        df = evaluate_videos(
            video_dir    = video_dir,
            latentsync_root = latentsync_root,
            output_path  = output_path,
            syncnet_ckpt = syncnet_ckpt,
            device       = device,
            batch_size   = batch_size,
            vshift       = vshift,
        )
        if df is not None and not df.empty:
            summary_rows.append({
                "dataset":         label,
                "n_videos":        len(df),
                "mean_confidence": round(float(df["confidence"].mean()), 4),
                "mean_offset_sec": round(float(df["offset_sec"].mean()), 4),
                "mean_abs_offset": round(float(df["offset_sec"].abs().mean()), 4),
                "mean_sync_score": round(float(df["sync_score"].mean()), 4),
            })

    summary_path = os.path.join(results_dir, "syncnet_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
