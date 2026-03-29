![AVBench Title](title.png)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E?style=for-the-badge)](https://huggingface.co/iiiiii123/AVBench_model)
[![Hugging Face Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-1D9BF0?style=for-the-badge)](https://huggingface.co/spaces/iiiiii123/AVBenchLB)
[![Project Website](https://img.shields.io/badge/%F0%9F%8C%90%20Project-Website-22A699?style=for-the-badge)](https://yajialiang.github.io/AVBench-site/)

# AVBench

Open-source benchmark workspace for video/audio quality and alignment evaluation.

## Repository Layout

- `video_data/`: Input videos to be evaluated.
	- You can put videos directly under this folder, or split them into subfolders (recommended).
- `dataset/`: Text/prompt datasets used by content-accuracy evaluation.
- `evaluation/`: Evaluation scripts.
- `results/`: Generated evaluation outputs (CSV/JSON).

## Evaluation Scripts

- `evaluation/evaluate_aesthetics.py`: Visual aesthetics (frame-based predictor).
- `evaluation/evaluate_dover.py`: Video quality with DOVER++ (aesthetic/technical/overall).
- `evaluation/evaluate_syncnet.py`: Lip-sync quality with SyncNet.
- `evaluation/evaluate_audiobox_aesthetics.py`: Audio aesthetics with Audiobox metrics.
- `evaluation/evaluate_nisqa.py`: Speech quality/naturalness with NISQA.
- `evaluation/evaluate_DF_arena_from_videos.py`: Anti-spoofing detection with DF_Arena.
- `evaluation/evaluate_speech_content_accuracy.py`: ASR-based speech content accuracy.
- `evaluation/evaluate_vt_consistency.py`: Video-text consistency evaluation.
- `evaluation/evaluate_at_consistency.py`: Audio-text consistency evaluation.
- `evaluation/evaluate_av_consistency.py`: Audio-video consistency evaluation.

## Model Download

- [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E?style=flat-square)](https://huggingface.co/iiiiii123/AVBench_model)

## Hugging Face Leaderboard

- [![Hugging Face Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-1D9BF0?style=flat-square)](https://huggingface.co/spaces/iiiiii123/AVBenchLB)

## Paper (arXiv)

- AVBench paper: ArXiv link will be updated here once the public arXiv page is available.

## Open-Source Plan

- Full blueprint (VBench-style): [OPEN_SOURCE_BLUEPRINT.md](OPEN_SOURCE_BLUEPRINT.md)

## Environment Requirements

- OS: Ubuntu 20.04/22.04 (recommended)
- Python: 3.10 (recommended)
- CUDA: 11.8 or 12.1
- PyTorch: 2.1+
- Required system tool: `ffmpeg`

Quick setup:

```bash
conda create -n avbench python=3.10 -y
conda activate avbench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate tqdm numpy pandas librosa soundfile qwen-vl-utils
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Video Naming Convention

- Canonical filename: `<sample_id>_<model_suffix>.mp4`
- `sample_id` must match one dataset identifier (`id`, `video_id`, `hash`, or `video_file` stem).
- `model_suffix` should be lowercase and stable across runs, e.g., `generated`, `qwen25omni`.
- Recommended folder layout: `video_data/<model_name>/`

Examples:

```text
video_data/my_model_v1/a13f9c2b_generated.mp4
video_data/my_model_v1/000123_generated.mp4
```

All scripts now:

- Default to AVBench-relative paths.
- Auto-discover datasets under `video_data/`.
- Save outputs to `results/`.
- Support CLI overrides (e.g., `--video-root`, `--results-dir`).

## Quick Start

Run from repository root:

```bash
python evaluation/evaluate_aesthetics.py
python evaluation/evaluate_dover.py
python evaluation/evaluate_syncnet.py
python evaluation/evaluate_audiobox_aesthetics.py
python evaluation/evaluate_nisqa.py
python evaluation/evaluate_DF_arena_from_videos.py
python evaluation/evaluate_speech_content_accuracy.py
python evaluation/evaluate_vt_consistency.py
python evaluation/evaluate_at_consistency.py
python evaluation/evaluate_av_consistency.py
```

Example with custom paths:

```bash
python evaluation/evaluate_dover.py \
	--video-root ./video_data \
	--results-dir ./results \
	--dover-root /path/to/DOVER
```