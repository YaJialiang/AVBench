#!/usr/bin/env python3
"""Evaluate audio-text consistency on AVBench normal subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common_paths import default_dataset_root, default_results_root, default_video_root


MODEL_HUB_URL = "https://huggingface.co/iiiiii123/AVBench_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate audio-text consistency on AVBench normal subset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/public/yangjl/LlamaFactory/saves/Qwen2-Audio-7B/full/train_2026-02-18-10-18-08",
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
        "--output-path",
        type=str,
        default=str(default_results_root() / "at_consistency_normal.json"),
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


def extract_audio(video_path: Path) -> str | None:
    """Extract 16k mono WAV audio from a video and return temporary file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
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
        tmp.name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp.name
    except subprocess.CalledProcessError:
        os.unlink(tmp.name)
        return None


class AudioTextConsistencyScorer:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        print("=" * 80)
        print("Audio-Text Consistency Evaluation")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Model hub: {MODEL_HUB_URL}\n")

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"Yes token id={self.yes_token_id}, No token id={self.no_token_id}\n")

    def get_consistency_score(self, audio_path: str, text_prompt: str, return_details: bool = False):
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            text_input = (
                "<|audio_bos|><|AUDIO|><|audio_eos|>\\n"
                f"Audio Description: {text_prompt}\\n\\n"
                "Does this audio description accurately match the audio content? "
                "Answer only Yes or No."
            )

            with torch.no_grad():
                inputs = self.processor(
                    text=[text_input],
                    audio=[audio],
                    sampling_rate=16000,
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

            first_logits = outputs.scores[0][0]
            yes_logit = first_logits[self.yes_token_id].item()
            no_logit = first_logits[self.no_token_id].item()

            probs = F.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
            yes_prob = probs[0].item()
            no_prob = probs[1].item()

            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if return_details:
                return yes_prob, {
                    "yes_probability": yes_prob,
                    "no_probability": no_prob,
                    "generated_text": generated_text,
                }
            return yes_prob
        except Exception as exc:  # pylint: disable=broad-except
            if return_details:
                return 0.0, {
                    "yes_probability": 0.0,
                    "no_probability": 1.0,
                    "generated_text": "",
                    "error": str(exc),
                }
            return 0.0


def main():
    args = parse_args()
    dataset_json = Path(args.dataset_json)
    video_dir = Path(args.video_dir)
    output_path = Path(args.output_path)

    with dataset_json.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Dataset size: {len(dataset)}")

    available: list[Tuple[Dict, Path]] = [
        (item, resolve_video_path(item, video_dir, args.model_suffix)) for item in dataset
    ]
    skipped = [(item, p) for item, p in available if not p.exists()]
    to_eval = [(item, p) for item, p in available if p.exists()]

    print(f"Evaluable: {len(to_eval)} | Skipped (missing video): {len(skipped)}")

    scorer = AudioTextConsistencyScorer(args.model_path)

    results = []
    scores = []

    for idx, (item, vp) in enumerate(tqdm(to_eval, desc="Evaluating")):
        prompt_audio = item.get("prompt_audio", "")
        utterance = item.get("utterance", "")

        if utterance:
            text_prompt = f'Audio Observation: {prompt_audio} Speech Content: "{utterance}"'
        else:
            text_prompt = prompt_audio

        tmp_audio = extract_audio(vp)
        if tmp_audio is None:
            sample_id = resolve_sample_id(item)
            results.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "video_file": item.get("video_file", f"{sample_id}.mp4"),
                    "generated_video": str(vp),
                    "text_prompt": text_prompt,
                    "consistency_score": 0.0,
                    "yes_probability": 0.0,
                    "no_probability": 1.0,
                    "generated_text": "",
                    "error": "audio extraction failed",
                }
            )
            continue

        try:
            score, details = scorer.get_consistency_score(tmp_audio, text_prompt, return_details=True)
            sample_id = resolve_sample_id(item)
            results.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "video_file": item.get("video_file", f"{sample_id}.mp4"),
                    "generated_video": str(vp),
                    "prompt_audio": prompt_audio,
                    "utterance": utterance,
                    "text_prompt": text_prompt,
                    "consistency_score": score,
                    "yes_probability": details["yes_probability"],
                    "no_probability": details["no_probability"],
                    "generated_text": details["generated_text"],
                }
            )
            scores.append(score)
        finally:
            if os.path.exists(tmp_audio):
                os.unlink(tmp_audio)

    avg = sum(scores) / len(scores) if scores else 0.0
    above_05 = sum(1 for s in scores if s >= 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": {
            "model_hub": MODEL_HUB_URL,
            "total_evaluated": len(results),
            "total_skipped": len(skipped),
            "skipped_files": [resolve_sample_id(item) for item, _ in skipped],
            "average_score": avg,
            "above_05_count": above_05,
            "above_05_ratio": above_05 / len(scores) if scores else 0.0,
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
