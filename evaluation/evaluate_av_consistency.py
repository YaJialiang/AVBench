#!/usr/bin/env python3
"""Evaluate audio-video consistency on AVBench normal subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common_paths import default_dataset_root, default_results_root, default_video_root

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils") from exc


MODEL_HUB_URL = "https://huggingface.co/iiiiii123/AVBench_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate audio-video consistency on AVBench normal subset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/public/yangjl/Qwen2.5-Omni-7B-AudioVideoMatching-Merged",
        help="Model path or HF snapshot directory.",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=str(default_dataset_root() / "normal_subset.json"),
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(default_video_root() / "normal"),
        help="Directory containing generated videos.",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(default_video_root() / "normal" / "generate_state.json"),
        help="Optional generation state JSON. Non-done samples will be skipped.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(default_results_root() / "av_consistency_normal.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="generated",
        help="Generated filename suffix: <sample_id>_<model_suffix>.mp4",
    )
    return parser.parse_args()


def resolve_sample_id(item: Dict) -> str:
    if item.get("video_file"):
        return Path(str(item["video_file"])).stem
    for key in ("id", "video_id", "hash"):
        if item.get(key):
            return str(item[key])
    return ""


def resolve_video_path(item: Dict, video_dir: Path, model_suffix: str) -> Path:
    sample_id = resolve_sample_id(item)
    return video_dir / f"{sample_id}_{model_suffix}.mp4"


def get_audio_sample_rate(video_path: str | Path) -> int:
    """Read source audio sample rate from video metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "a",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout or "{}")
        streams = info.get("streams", [])
        if streams:
            return int(streams[0].get("sample_rate", 0))
    except Exception:  # pylint: disable=broad-except
        pass
    return 0


def extract_audio_from_video(video_path: str | Path, output_audio_path: str | None = None) -> str | None:
    """Extract mono 16k WAV audio from a video file."""
    if output_audio_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_audio_path = tmp.name
        tmp.close()

    src_sr = get_audio_sample_rate(video_path)

    if src_sr > 48000:
        # Two-step resampling is more stable for very high sample-rate sources.
        mid_path = tempfile.mktemp(suffix="_mid.wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-y",
                    mid_path,
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    mid_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    str(output_audio_path),
                ],
                check=True,
                capture_output=True,
            )
            return output_audio_path
        except subprocess.CalledProcessError:
            pass
        finally:
            if os.path.exists(mid_path):
                os.unlink(mid_path)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(output_audio_path),
            ],
            check=True,
            capture_output=True,
        )
        return output_audio_path
    except subprocess.CalledProcessError:
        return None


class AudioVideoConsistencyScorer:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

        print("=" * 80)
        print("Audio-Video Consistency Evaluation")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Model hub: {MODEL_HUB_URL}\n")

        try:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to load model/processor: {exc}") from exc

        self.tokenizer = self.processor.tokenizer
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"Yes token id={self.yes_token_id}, No token id={self.no_token_id}\n")

    def get_consistency_score(
        self,
        video_path: str,
        audio_path: str | None = None,
        return_details: bool = False,
    ):
        temp_audio_file = None
        if audio_path is None or audio_path == video_path:
            extracted_audio = extract_audio_from_video(video_path)
            if extracted_audio is None:
                if return_details:
                    return 0.0, {"error": "No audio stream found"}
                return 0.0
            audio_path = extracted_audio
            temp_audio_file = extracted_audio

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "fps": 1.0,
                            "max_pixels": 480 * 640,
                        },
                        {"type": "audio", "audio": str(audio_path)},
                        {"type": "text", "text": "Is this audio and video properly matched?"},
                    ],
                }
            ]

            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

            try:
                audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
                max_audio_samples = int(9.9 * sr)
                if len(audio_data) > max_audio_samples:
                    audio_data = audio_data[:max_audio_samples]
            except Exception as exc:  # pylint: disable=broad-except
                if return_details:
                    return 0.0, {"error": f"Audio load failed: {exc}"}
                return 0.0

            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            with torch.no_grad():
                inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    audio=[audio_data],
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            if hasattr(outputs, "scores") and len(outputs.scores) > 0:
                first_logits = outputs.scores[0][0]
                yes_logit = first_logits[self.yes_token_id].item()
                no_logit = first_logits[self.no_token_id].item()
                probs = F.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
                yes_prob = probs[0].item()
                no_prob = probs[1].item()
                score = yes_prob
            else:
                score = 0.0
                yes_prob, no_prob = 0.0, 0.0

            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if return_details:
                return score, {
                    "score": score,
                    "yes_probability": yes_prob,
                    "no_probability": no_prob,
                    "generated_text": generated_text,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                }
            return score
        finally:
            if temp_audio_file is not None and os.path.exists(temp_audio_file):
                try:
                    os.unlink(temp_audio_file)
                except OSError:
                    pass


def load_skip_files(state_path: Path) -> set[str]:
    """Read failed/non-done samples from generate_state.json."""
    if not state_path.exists():
        print(f"State file not found: {state_path}. No state-based skipping.")
        return set()

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    failed = {k for k, v in state.items() if v.get("status") != "done"}
    print(f"Loaded {len(failed)} non-done files from state. They will be skipped.")
    return failed


def main():
    args = parse_args()
    dataset_json = Path(args.dataset_json)
    video_dir = Path(args.video_dir)
    state_path = Path(args.state_path)
    output_path = Path(args.output_path)

    with dataset_json.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Dataset size: {len(dataset)}")

    skip_set = load_skip_files(state_path)

    # Keep schema compatibility for both legacy and AVBench datasets.
    before = len(dataset)
    dataset = [
        item
        for item in dataset
        if item.get("video_file", f"{resolve_sample_id(item)}.mp4") not in skip_set
    ]
    print(f"State-skipped: {before - len(dataset)} | Remaining: {len(dataset)}")

    to_eval: list[Tuple[Dict, Path]] = [
        (item, resolve_video_path(item, video_dir, args.model_suffix)) for item in dataset
    ]
    missing = [(item, p) for item, p in to_eval if not p.exists()]
    to_eval = [(item, p) for item, p in to_eval if p.exists()]

    if missing:
        print(f"Missing-video skipped: {len(missing)}")

    print(f"Actual evaluations: {len(to_eval)}")

    scorer = AudioVideoConsistencyScorer(args.model_path)

    results = []
    scores = []

    for idx, (item, vp) in enumerate(tqdm(to_eval, desc="Evaluating")):
        score, details = scorer.get_consistency_score(str(vp), return_details=True)

        sample_id = resolve_sample_id(item)
        entry = {
            "index": idx,
            "id": sample_id,
            "video_file": item.get("video_file", f"{sample_id}.mp4"),
            "generated_video": str(vp),
            "consistency_score": score,
            "yes_probability": details.get("yes_probability", 0.0),
            "no_probability": details.get("no_probability", 1.0),
            "generated_text": details.get("generated_text", ""),
        }
        if "error" in details:
            entry["error"] = details["error"]

        results.append(entry)
        scores.append(score)

    valid_scores = [s for s in scores if s is not None]
    avg = float(np.mean(valid_scores)) if valid_scores else 0.0
    std = float(np.std(valid_scores)) if valid_scores else 0.0
    above_05 = sum(1 for s in valid_scores if s >= 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model_hub": MODEL_HUB_URL,
        "model_path": args.model_path,
        "dataset_json": str(dataset_json),
        "video_dir": str(video_dir),
        "state_path": str(state_path),
        "skipped_files_from_state": sorted(skip_set),
        "total_samples": len(results),
        "successful_evaluations": len(valid_scores),
        "statistics": {
            "mean": avg,
            "std": std,
            "min": float(min(valid_scores)) if valid_scores else 0.0,
            "max": float(max(valid_scores)) if valid_scores else 0.0,
            "above_05_count": above_05,
            "above_05_ratio": above_05 / len(valid_scores) if valid_scores else 0.0,
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
