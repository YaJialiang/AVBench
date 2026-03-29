#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate AVBench video audio quality with Audiobox Aesthetics metrics."""

import os
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import subprocess
import tempfile
import json
import warnings
from common_paths import default_results_root, default_video_root, discover_video_dirs

warnings.filterwarnings('ignore')


def find_video_files(video_dir: str) -> List[str]:
    """Find all video files under a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    return sorted([str(f) for f in video_files])


def extract_video_id(video_path: str) -> str:
    """Extract ID from video filename"""
    filename = Path(video_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return filename


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """
    Extract audio from video
    
    Args:
        video_path: Video file path
        output_dir: Output directory
        
    Returns:
        Extracted audio file path
    """
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    
    # Extract audio with ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # Disable video stream
        '-acodec', 'pcm_s16le',  # Use PCM codec
        '-ar', '16000',  # Sample rate 16kHz
        '-ac', '1',  # Mono channel
        '-y',  # Overwrite existing file
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr}")


def evaluate_audio_aesthetics(audio_paths: List[str], 
                              checkpoint_path: str = None,
                              batch_size: int = 16) -> List[Dict[str, float]]:
    """
    Evaluate audio quality with Audiobox Aesthetics
    
    Args:
        audio_paths: List of audio file paths
        checkpoint_path: Model checkpoint path
        batch_size: Batch size
        
    Returns:
        Evaluation results list
    """
    # Create input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        input_file = f.name
        for audio_path in audio_paths:
            f.write(json.dumps({"path": audio_path}) + '\n')
    
    try:
        # Build command
        cmd = ['audio-aes', input_file, '--batch-size', str(batch_size)]
        
        if checkpoint_path:
            cmd.extend(['--ckpt', checkpoint_path])
        
        # Run evaluation
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        # Parse outputs
        results = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                results.append(json.loads(line))
        
        return results
        
    finally:
        # Clean temporary files
        if os.path.exists(input_file):
            os.remove(input_file)


def evaluate_videos(video_dir: str,
                    output_path: str,
                    checkpoint_path: str = None,
                    batch_size: int = 16,
                    keep_audio: bool = False):
    """
    Evaluate audio aesthetics for all videos
    
    Args:
        video_dir: Video directory
        output_path: Output CSV path
        checkpoint_path: AudioboxModel checkpoint path
        batch_size: Batch size
        keep_audio: Whether to keep extracted audio files
    """
    print("=" * 80)
    print("Audiobox Aesthetics Audio Quality Evaluation")
    print("=" * 80)
    
    # 查找所有视频文件
    video_files = find_video_files(video_dir)
    print(f"\nFound {len(video_files)} video files")
    
    if not video_files:
        print(f"Error: no videos found in {video_dir}")
        return
    
    # Create temporary directory for audio
    audio_dir = tempfile.mkdtemp(prefix='va_bench_audio_')
    print(f"\nAudio extraction directory: {audio_dir}")
    
    try:
        # Step 1/2: Extract audio from videos
        print("\nStep 1/2: Extracting audio from videos...")
        audio_paths = []
        video_audio_map = {}
        
        for video_path in tqdm(video_files, desc="Extract audio"):
            try:
                audio_path = extract_audio_from_video(video_path, audio_dir)
                audio_paths.append(audio_path)
                video_audio_map[audio_path] = video_path
            except Exception as e:
                print(f"\nWarning: failed to extract audio from {Path(video_path).name}: {str(e)}")
        
        if not audio_paths:
            print("Error: no audio files were extracted successfully")
            return
        
        print(f"Successfully extracted {len(audio_paths)} audio files")
        
        # Step 2/2: Evaluate audio aesthetics
        print(f"\nStep 2/2: Evaluating audio aesthetics...")
        print(f"Using Audiobox Aesthetics model...")
        print(f"Metrics: CE (Content Enjoyment), CU (Content Usefulness), PC (Production Complexity), PQ (Production Quality)")
        
        aesthetics_results = evaluate_audio_aesthetics(
            audio_paths, 
            checkpoint_path=checkpoint_path,
            batch_size=batch_size
        )
        
        # Aggregate results
        results = []
        for audio_path, aesthetics in zip(audio_paths, aesthetics_results):
            video_path = video_audio_map[audio_path]
            video_id = extract_video_id(video_path)
            video_name = Path(video_path).name
            
            results.append({
                'video_id': video_id,
                'video_name': video_name,
                'content_enjoyment': aesthetics.get('CE', None),
                'content_usefulness': aesthetics.get('CU', None),
                'production_complexity': aesthetics.get('PC', None),
                'production_quality': aesthetics.get('PQ', None),
                # Paper formula: S = (CE + CU + PQ - PC) / 4
                'avg_score': (
                    aesthetics.get('CE', 0)
                    + aesthetics.get('CU', 0)
                    + aesthetics.get('PQ', 0)
                    - aesthetics.get('PC', 0)
                ) / 4
            })
        
        # Save results
        df = pd.DataFrame(results)
        df = df.sort_values('video_id')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nEvaluation completed. Results saved to: {output_path}")
        return df
        
    finally:
        if not keep_audio:
            print(f"\nCleaning temporary audio files...")
            import shutil
            if os.path.exists(audio_dir):
                shutil.rmtree(audio_dir)
        else:
            print(f"\nKeeping audio files at: {audio_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate audio aesthetics with Audiobox Aesthetics.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--checkpoint", default="/public/yangjl/audiobox-aesthetics/checkpoint.pt", help="Audiobox checkpoint path (optional).")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    video_dirs = discover_video_dirs(Path(args.video_root))
    if not os.path.exists(checkpoint_path):
        print(f"Warning: checkpoint not found: {checkpoint_path}. Falling back to default model.")
        checkpoint_path = None

    # Check whether audio-aes command is available
    try:
        subprocess.run(['audio-aes', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'audio-aes' command not found.\nInstall with: pip install audiobox-aesthetics")
        return

    batch_size  = args.batch_size
    results_dir = args.results_dir
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
        output_path = os.path.join(results_dir, f"audiobox_{label}.csv")
        df = evaluate_videos(
            video_dir       = video_dir,
            output_path     = output_path,
            checkpoint_path = checkpoint_path,
            batch_size      = batch_size,
            keep_audio      = False,
        )
        if df is not None and not df.empty:
            summary_rows.append({
                "dataset":         label,
                "n_videos":        len(df),
                "mean_CE":         round(float(df["content_enjoyment"].mean()),    4),
                "mean_CU":         round(float(df["content_usefulness"].mean()),   4),
                "mean_PC":         round(float(df["production_complexity"].mean()),4),
                "mean_PQ":         round(float(df["production_quality"].mean()),   4),
                # (CE + CU + PQ - PC) / 4
                "mean_avg_score":  round(float(df["avg_score"].mean()),            4),
            })

    summary_path = os.path.join(results_dir, "audiobox_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
