#!/usr/bin/env python3
"""
Extract audio from AVBench videos and run DF_Arena anti-spoofing evaluation.
"""
import sys
import os
import argparse

# Check runtime environment first
try:
    from transformers import pipeline
    import librosa
    import json
    from pathlib import Path
    from tqdm import tqdm
    import subprocess
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install required dependencies (transformers, librosa, tqdm) and rerun.")
    sys.exit(1)

from common_paths import default_results_root, default_video_root, discover_video_dirs

TEMP_AUDIO_DIR = "/tmp/df_arena_audio_extracted"


def evaluate_one_dir(video_dir: str, output_path: str, pipe) -> dict:
    """Evaluate one video directory and return summary statistics."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(video_dir).glob(f"*{ext}")))
    video_files = sorted(video_files)
    print(f"Found {len(video_files)} video files")

    if not video_files:
        return {}

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    results = []
    spoof_count = bonafide_count = error_count = extraction_errors = 0
    # Collect bonafide probabilities for the average score.
    bonafide_scores = []

    for idx, video_path in enumerate(tqdm(video_files, desc="Processing")):
        temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"{video_path.stem}.wav")
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ar', '16000', '-ac', '1', '-y',
                '-loglevel', 'error',
                temp_audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ffmpeg extraction failed: {result.stderr}")

            audio, sr = librosa.load(temp_audio_path, sr=16000)
            detection_result = pipe(audio)

            bonafide_score = detection_result['all_scores']['bonafide']
            bonafide_scores.append(bonafide_score)

            if detection_result['label'] == 'spoof':
                spoof_count += 1
            else:
                bonafide_count += 1

            results.append({
                "video_filename": video_path.name,
                "label":          detection_result['label'],
                "score":          detection_result['score'],
                "spoof_score":    detection_result['all_scores']['spoof'],
                "bonafide_score": bonafide_score,
            })
        except Exception as e:
            error_count += 1
            if "ffmpeg" in str(e).lower():
                extraction_errors += 1
            results.append({"video_filename": video_path.name, "error": str(e)})
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass

    total = len(video_files)
    valid = len(bonafide_scores)
    avg_bonafide_score = sum(bonafide_scores) / valid if valid > 0 else 0.0
    summary = {
        "video_dir":           video_dir,
        "total":               total,
        "valid":               valid,
        "spoof_count":         spoof_count,
        "bonafide_count":      bonafide_count,
        "error_count":         error_count,
        "extraction_errors":   extraction_errors,
        "spoof_ratio":         spoof_count  / valid if valid > 0 else 0,
        "bonafide_ratio":      bonafide_count / valid if valid > 0 else 0,
        "avg_bonafide_score":  avg_bonafide_score,
        "results":             results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  avg_bonafide_score: {avg_bonafide_score:.4f}  "
          f"(spoof={spoof_count}, bonafide={bonafide_count}, error={error_count}/{total})")
    print(f"  Results saved: {output_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run DF_Arena anti-spoofing on AVBench videos.")
    parser.add_argument(
        "--video-root",
        type=str,
        default=str(default_video_root()),
        help="Root directory of video datasets. Subfolders are discovered automatically.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results_root()),
        help="Directory to save per-dataset and summary outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    video_dirs = discover_video_dirs(args.video_root)

    print("="*80)
    print("DF Arena anti-spoofing evaluation - multi-directory batch mode")
    print("="*80)
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"No evaluable video directories found under: {args.video_root}")
        return

    print("\nLoading DF_Arena model...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    pipe = pipeline("antispoofing", model="Speech-Arena-2025/DF_Arena_1B_V_1",
                    trust_remote_code=True, device='cuda', local_files_only=True)
    print("✅ Model loaded")

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*80}")
        print(f"# Dataset: {label}  ({video_dir})")
        print(f"{'#'*80}")
        if not os.path.exists(video_dir):
            print("Skipping: directory does not exist")
            continue
        output_path = os.path.join(results_dir, f"df_arena_{label}.json")
        stat = evaluate_one_dir(video_dir, output_path, pipe)
        if stat:
            summary_rows.append({
                "dataset":            label,
                "total":              stat["total"],
                "valid":              stat.get("valid", stat["total"]),
                "spoof_count":        stat["spoof_count"],
                "bonafide_count":     stat["bonafide_count"],
                "error_count":        stat["error_count"],
                "spoof_ratio":        round(stat["spoof_ratio"], 4),
                "bonafide_ratio":     round(stat["bonafide_ratio"], 4),
                "avg_bonafide_score": round(stat["avg_bonafide_score"], 4),
            })

    summary_path = os.path.join(results_dir, "df_arena_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*80}")
    for r in summary_rows:
        print(f"  {r['dataset']:20s}: avg_bonafide={r['avg_bonafide_score']:.4f}  "
              f"spoof={r['spoof_ratio']*100:5.1f}%  "
              f"bonafide={r['bonafide_ratio']*100:5.1f}%  "
              f"valid={r['valid']}/{r['total']}  error={r['error_count']}")


if __name__ == "__main__":
    main()

