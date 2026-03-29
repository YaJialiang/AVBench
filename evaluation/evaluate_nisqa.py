#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate speech quality/naturalness of AVBench videos with NISQA."""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import tempfile
import subprocess
from common_paths import default_results_root, default_video_root, discover_video_dirs

warnings.filterwarnings('ignore')

def load_nisqa_model_class(nisqa_root: str):
    """Lazy import NISQA class from a configurable repository root."""
    sys.path.insert(0, nisqa_root)
    from nisqa.NISQA_model import nisqaModel
    return nisqaModel


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


def extract_audio_from_video(video_path: str, output_dir: str = None) -> str:
    """
    Extract mono WAV audio from a video file.
    
    Args:
        video_path: Video file path
        output_dir: Output directory. If None, a temp directory is used.
        
    Returns:
        Extracted audio file path
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    
    # Extract audio with ffmpeg
    # NISQA commonly uses 16kHz or 48kHz audio.
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
    
    subprocess.run(
        cmd, 
        check=True, 
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    return audio_path


def evaluate_videos(video_dir: str,
                    output_path: str,
                    model_path: str,
                    nisqa_root: str,
                    use_tts_model: bool = False):
    """Evaluate speech quality for all videos in a directory."""
    print("=" * 80)
    print("NISQA v2.0 speech-quality evaluation")
    print("=" * 80)
    
    if use_tts_model:
        print("\nModel type: NISQA-TTS (synthetic speech naturalness)")
        print("Metrics: Naturalness")
    else:
        print("\nModel type: NISQA v2.0 (transmitted speech quality)")
        print("Metrics: MOS, Noisiness, Coloration, Discontinuity, Loudness")
    
    # Discover input videos.
    video_files = find_video_files(video_dir)
    print(f"\nFound {len(video_files)} video files")
    
    if not video_files:
        print(f"Error: no videos found in {video_dir}")
        return
    
    # Create temporary directory for extracted audio.
    audio_dir = tempfile.mkdtemp(prefix='va_bench_nisqa_')
    print(f"\nAudio extraction directory: {audio_dir}")
    
    try:
        # Step 1/3: extract audio.
        print(f"\nStep 1/3: Extracting audio from videos...")
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
        
        # Step 2/3: prepare NISQA CSV input.
        print(f"\nStep 2/3: Preparing NISQA input...")
        csv_input_path = os.path.join(audio_dir, 'nisqa_input.csv')
        df_input = pd.DataFrame({
            'filepath': audio_paths
        })
        df_input.to_csv(csv_input_path, index=False)
        print(f"Input CSV created: {csv_input_path}")
        
        # Step 3/3: run NISQA prediction.
        print(f"\nStep 3/3: Running NISQA evaluation...")
        print(f"This may take several minutes...")
        
        # Configure NISQA inference arguments.
        args = {
            'mode': 'predict_csv',
            'pretrained_model': model_path,
            'csv_file': csv_input_path,
            'csv_deg': 'filepath',
            'data_dir': '',
            'output_dir': audio_dir,
            'num_workers': 0,
            'bs': 4,  # Batch size
            'tr_bs_val': 4,
            'tr_num_workers': 0,
            'ms_channel': None
        }
        
        # Initialize and run NISQA.
        nisqaModel = load_nisqa_model_class(nisqa_root)
        nisqa = nisqaModel(args)
        nisqa.predict()
        
        # Read NISQA output.
        nisqa_output_path = os.path.join(audio_dir, 'NISQA_results.csv')
        if not os.path.exists(nisqa_output_path):
            print(f"Error: NISQA output file not found: {nisqa_output_path}")
            return
        
        df_nisqa = pd.read_csv(nisqa_output_path)
        print(f"✓ NISQA evaluation completed, processed {len(df_nisqa)} audio files")
        
        # Aggregate results
        print(f"\nAggregating evaluation results...")
        results = []
        
        for idx, row in df_nisqa.iterrows():
            audio_path = row['filepath']
            if audio_path in video_audio_map:
                video_path = video_audio_map[audio_path]
                video_id = extract_video_id(video_path)
                video_name = Path(video_path).name
                
                result = {
                    'video_id': video_id,
                    'video_name': video_name,
                }
                
                if use_tts_model:
                    # TTS mode outputs naturalness only.
                    result['naturalness'] = float(row.get('mos_pred', row.get('noi_pred', 0)))
                else:
                    # NISQA v2.0 outputs multiple dimensions.
                    mos = float(row.get('mos_pred', 0))
                    result['mos'] = mos
                    result['noisiness'] = float(row.get('noi_pred', 0))
                    result['coloration'] = float(row.get('col_pred', 0))
                    result['discontinuity'] = float(row.get('dis_pred', 0))
                    result['loudness'] = float(row.get('loud_pred', 0))
                    
                    # Add quality grade from MOS.
                    if mos >= 4.0:
                        result['quality_grade'] = 'Excellent'
                    elif mos >= 3.5:
                        result['quality_grade'] = 'Good'
                    elif mos >= 3.0:
                        result['quality_grade'] = 'Fair'
                    else:
                        result['quality_grade'] = 'Poor'
                
                results.append(result)
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('video_id').reset_index(drop=True)
            df_results.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"\n{'=' * 80}")
            print("Evaluation completed")
            print(f"{'=' * 80}")
            
            print(f"\nSuccessfully evaluated videos: {len(results)}/{len(video_files)}")
            print(f"Results saved to: {output_path}")
            
            # Summary statistics.
            print(f"\n{'=' * 80}")
            print("Statistics (score range: 1-5, higher is better)")
            print(f"{'=' * 80}")
            
            if use_tts_model:
                print(f"\nNaturalness statistics:")
                nat_scores = df_results['naturalness'].values
                print(f"  Mean: {nat_scores.mean():.4f}")
                print(f"  Std: {nat_scores.std():.4f}")
                print(f"  Min: {nat_scores.min():.4f}")
                print(f"  Max: {nat_scores.max():.4f}")
                print(f"  Median: {np.median(nat_scores):.4f}")
                
                # Show highest/lowest naturalness examples.
                print(f"\nTop 10 videos by naturalness:")
                top_10 = df_results.nlargest(10, 'naturalness')
                for idx, row in top_10.iterrows():
                    print(f"  {row['video_id']}: {row['naturalness']:.3f}")
                
                print(f"\nBottom 10 videos by naturalness:")
                bottom_10 = df_results.nsmallest(10, 'naturalness')
                for idx, row in bottom_10.iterrows():
                    print(f"  {row['video_id']}: {row['naturalness']:.3f}")
            
            else:
                print(f"\nMOS (overall quality) statistics:")
                mos_scores = df_results['mos'].values
                print(f"  Mean: {mos_scores.mean():.3f}")
                print(f"  Std: {mos_scores.std():.3f}")
                print(f"  Min: {mos_scores.min():.3f}")
                print(f"  Max: {mos_scores.max():.3f}")
                print(f"  Median: {np.median(mos_scores):.3f}")
                
                print(f"\nMean score by quality dimension:")
                print(f"  Noisiness (NOI): {df_results['noisiness'].mean():.3f} (higher is less noisy)")
                print(f"  Coloration (COL): {df_results['coloration'].mean():.3f} (higher is more natural timbre)")
                print(f"  Discontinuity (DIS): {df_results['discontinuity'].mean():.3f} (higher is more continuous)")
                print(f"  Loudness (LOUD): {df_results['loudness'].mean():.3f} (higher is more appropriate loudness)")
                
                print(f"\nQuality-grade distribution:")
                grade_counts = df_results['quality_grade'].value_counts()
                for grade in ['Excellent', 'Good', 'Fair', 'Poor']:
                    count = grade_counts.get(grade, 0)
                    pct = count / len(df_results) * 100
                    print(f"  {grade}: {count} ({pct:.1f}%)")
                
                print(f"\nTop 10 videos by MOS:")
                best = df_results.nlargest(10, 'mos')
                for idx, row in best.iterrows():
                    print(f"  {row['video_id']}: MOS={row['mos']:.3f}, {row['quality_grade']}")
                
                print(f"\nBottom 10 videos by MOS:")
                worst = df_results.nsmallest(10, 'mos')
                for idx, row in worst.iterrows():
                    print(f"  {row['video_id']}: MOS={row['mos']:.3f}, {row['quality_grade']}")
                
                # Diagnose low-scoring dimensions.
                print(f"\nIssue analysis by quality dimension:")
                
                # Noisiness issues.
                noisy = df_results[df_results['noisiness'] < 3.0]
                if len(noisy) > 0:
                    print(f"  Noisiness issue (NOI<3.0): {len(noisy)} videos")
                    print(f"    Most severe: {noisy.nsmallest(3, 'noisiness')['video_id'].tolist()}")
                
                # Discontinuity issues.
                discontinuous = df_results[df_results['discontinuity'] < 3.0]
                if len(discontinuous) > 0:
                    print(f"  Discontinuity issue (DIS<3.0): {len(discontinuous)} videos")
                    print(f"    Most severe: {discontinuous.nsmallest(3, 'discontinuity')['video_id'].tolist()}")
                
                # Coloration issues.
                colored = df_results[df_results['coloration'] < 3.0]
                if len(colored) > 0:
                    print(f"  Coloration issue (COL<3.0): {len(colored)} videos")
                    print(f"    Most severe: {colored.nsmallest(3, 'coloration')['video_id'].tolist()}")
                
                # Loudness issues.
                loud_issue = df_results[df_results['loudness'] < 3.0]
                if len(loud_issue) > 0:
                    print(f"  Loudness issue (LOUD<3.0): {len(loud_issue)} videos")
                    print(f"    Most severe: {loud_issue.nsmallest(3, 'loudness')['video_id'].tolist()}")
        
        else:
            print(f"\n{'=' * 80}")
            print("Evaluation completed")
            print(f"{'=' * 80}")
            print("\nNo videos were evaluated successfully")
            df_results = pd.DataFrame()

    finally:
        print(f"\nCleaning temporary audio files...")
        import shutil
        try:
            shutil.rmtree(audio_dir)
        except Exception as e:
            print(f"Warning: failed to clean temporary directory: {e}")

    return df_results if 'df_results' in dir() else pd.DataFrame()


def parse_args():
    parser = argparse.ArgumentParser(description="Run NISQA speech quality evaluation.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--nisqa-root", default=os.environ.get("NISQA_ROOT", "/public/yangjl/NISQA"), help="Path to NISQA repository root.")
    parser.add_argument("--model-path", default=None, help="Path to NISQA checkpoint (.tar).")
    parser.add_argument("--use-tts-model", action="store_true", help="Use NISQA-TTS mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    nisqa_root = args.nisqa_root
    model_path = args.model_path or os.path.join(nisqa_root, "weights", "nisqa.tar")
    use_tts_model = args.use_tts_model
    results_dir = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"No evaluable video directories found under: {args.video_root}")
        return

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*80}")
        print(f"# Dataset: {label}  ({video_dir})")
        print(f"{'#'*80}")
        if not os.path.exists(video_dir):
            print("Skipping: directory does not exist")
            continue
        output_path = os.path.join(results_dir, f"nisqa_{label}.csv")
        df = evaluate_videos(
            video_dir     = video_dir,
            output_path   = output_path,
            model_path    = model_path,
            nisqa_root    = nisqa_root,
            use_tts_model = use_tts_model,
        )
        if df is not None and not df.empty:
            row = {"dataset": label, "n_videos": len(df)}
            if use_tts_model:
                row["mean_naturalness"] = round(float(df["naturalness"].mean()), 4)
            else:
                for col in ["mos", "noisiness", "coloration", "discontinuity", "loudness"]:
                    if col in df.columns:
                        row[f"mean_{col}"] = round(float(df[col].mean()), 4)
            summary_rows.append(row)

    summary_path = os.path.join(results_dir, "nisqa_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
