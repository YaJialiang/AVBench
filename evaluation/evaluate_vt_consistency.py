#!/usr/bin/env python3
"""Evaluate video-text consistency on AVBench normal subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from tqdm import tqdm

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils") from exc

from common_paths import default_dataset_root, default_results_root, default_video_root


MODEL_HUB_URL = "https://huggingface.co/iiiiii123/AVBench_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate video-text consistency on AVBench normal subset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/public/yangjl/Qwen2.5-Omni-7B-VideoTextMatching-Merged",
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
        default=str(default_results_root() / "vt_consistency_normal.json"),
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
    # Support both legacy schema (video_file) and AVBench schema (id/video_id/hash).
    if item.get("video_file"):
        return Path(str(item["video_file"])).stem
    for key in ("id", "video_id", "hash"):
        if item.get(key):
            return str(item[key])
    return ""


def resolve_video_path(item: Dict, video_dir: Path, model_suffix: str) -> Path:
    sample_id = resolve_sample_id(item)
    return video_dir / f"{sample_id}_{model_suffix}.mp4"


class VideoTextConsistencyScorer:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

        print("=" * 80)
        print("Video-Text Consistency Evaluation")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Model hub: {MODEL_HUB_URL}\n")

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
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

    def get_consistency_score(self, video_path: Path, text_prompt: str, return_details: bool = False):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "max_pixels": 360 * 420, "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Does the video match this description?\\n{text_prompt}\\nAnswer Yes or No.",
                        },
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: v.to(self.model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
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

            probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
            yes_prob = probs[0].item()
            no_prob = probs[1].item()

            new_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

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

    scorer = VideoTextConsistencyScorer(args.model_path)

    results = []
    scores = []

    for idx, (item, vp) in enumerate(tqdm(to_eval, desc="Evaluating")):
        prompt_vision = item.get("prompt_vision", "")
        score, details = scorer.get_consistency_score(vp, prompt_vision, return_details=True)

        sample_id = resolve_sample_id(item)
        entry = {
            "index": idx,
            "id": sample_id,
            "video_file": item.get("video_file", f"{sample_id}.mp4"),
            "generated_video": str(vp),
            "prompt_vision": prompt_vision,
            "consistency_score": score,
            "yes_probability": details.get("yes_probability", 0.0),
            "no_probability": details.get("no_probability", 1.0),
            "generated_text": details.get("generated_text", ""),
        }
        if "error" in details:
            entry["error"] = details["error"]
        results.append(entry)
        scores.append(score)

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
