"""Evaluate speech content fidelity (completeness/accuracy/hallucination) for AVBench videos."""

import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import warnings
import difflib
import re
import json
from common_paths import (
    default_dataset_root,
    default_results_root,
    default_video_root,
    discover_video_dirs,
)

warnings.filterwarnings('ignore')


def extract_audio_from_video(video_path: str, output_audio: str) -> bool:
    """Extract mono 16kHz WAV audio from a video file."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', output_audio
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return False


class SpeechContentEvaluator:
    """Speech content evaluator"""
    
    def __init__(self, asr_model="openai/whisper-large-v3", device="cuda"):
        """
        Initialize evaluator
        
        Args:
            asr_model: ASR model path or name
            device: Runtime device
        """
        self.device = device
        print("Loading ASR model...")
        
        # Import whisper
        try:
            import whisper
            self.whisper_model = whisper.load_model("large-v3", device=device)
            self.use_transformers = False
            print("✓ Using OpenAI Whisper model\n")
        except:
            # Fallback to transformers backend when whisper is unavailable
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                asr_model,
                torch_dtype="auto",
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(asr_model)
            
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                # The model already owns device mapping via device_map="auto".
            )
            self.use_transformers = True
            print(f"✓ Using Transformers Whisper backend\n")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Run ASR to transcribe audio
        
        Args:
            audio_path: Audio file path
            
        Returns:
            Transcribed text
        """
        try:
            if self.use_transformers:
                # language=None Auto language detection (multilingual datasets)
                result = self.asr_pipeline(audio_path, generate_kwargs={"language": None})
                return result['text'].strip()
            else:
                # language=None Auto language detection
                result = self.whisper_model.transcribe(audio_path, language=None)
                return result['text'].strip()
        except Exception as e:
            print(f"ASR failed: {e}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: remove punctuation and normalize spaces
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords with a simple stopword filter
        
        Args:
            text: Text
            
        Returns:
            Keyword list
        """
        # Simple Chinese stopword list for character-level matching.
        stopwords = set(['的', '了', '在', '是', '我', '你', '他', '她', '它', 
                        '们', '这', '那', '有', '个', '和', '与', '或', '但',
                        '啊', '呀', '吗', '呢', '吧', '嘛'])
        
        # Simple tokenization by character
        words = list(text)
        
        # Filter stopwords
        keywords = [w for w in words if w not in stopwords and w.strip()]
        
        return keywords
    
    def calculate_completeness(self, reference: str, hypothesis: str) -> float:
        """
        Compute completeness as keyword coverage from reference to hypothesis.

        Args:
            reference: Reference prompt text.
            hypothesis: ASR transcription text.

        Returns:
            Completeness score (0-100)
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # Extract keywords
        ref_keywords = self.extract_keywords(ref_norm)
        
        if not ref_keywords:
            return 100.0
        
        # Count covered keywords
        covered = sum(1 for kw in ref_keywords if kw in hyp_norm)
        completeness = (covered / len(ref_keywords)) * 100
        
        return completeness
    
    def calculate_accuracy(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute accuracy using multiple similarity metrics
        
        Args:
            reference: Reference prompt text.
            hypothesis: ASR transcription text.
            
        Returns:
            Dictionary of accuracy metrics
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # 1. Character-level edit-distance similarity
        edit_distance_ratio = difflib.SequenceMatcher(None, ref_norm, hyp_norm).ratio()
        
        # 2. Word-level overlap
        ref_words = set(ref_norm.split())
        hyp_words = set(hyp_norm.split())
        
        if not ref_words:
            word_overlap = 1.0
        else:
            word_overlap = len(ref_words & hyp_words) / len(ref_words)
        
        # 3. Character-level overlap
        ref_chars = set(ref_norm)
        hyp_chars = set(hyp_norm)
        
        if not ref_chars:
            char_overlap = 1.0
        else:
            char_overlap = len(ref_chars & hyp_chars) / len(ref_chars)
        
        # Weighted overall accuracy
        accuracy = (edit_distance_ratio * 0.5 + word_overlap * 0.3 + char_overlap * 0.2) * 100
        
        return {
            'edit_distance_similarity': edit_distance_ratio * 100,
            'word_overlap': word_overlap * 100,
            'char_overlap': char_overlap * 100,
            'accuracy': accuracy
        }
    
    def calculate_hallucination(self, reference: str, hypothesis: str) -> float:
        """
        Compute no-hallucination score based on extra content in hypothesis.

        Args:
            reference: Reference prompt text.
            hypothesis: ASR transcription text.

        Returns:
            Hallucination score (0-100), 100 means no extra content
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # Extract keywords from ASR hypothesis.
        hyp_keywords = self.extract_keywords(hyp_norm)
        
        if not hyp_keywords:
            return 100.0
        
        # Count tokens appearing in hypothesis but not in reference.
        hallucinated = sum(1 for kw in hyp_keywords if kw not in ref_norm)
        hallucination_rate = hallucinated / len(hyp_keywords)
        
        # Convert to score (fewer hallucinations is better)
        hallucination_score = (1 - hallucination_rate) * 100
        
        return hallucination_score
    
    def evaluate_content(self, reference: str, hypothesis: str) -> Dict:
        """
        Comprehensive speech-content evaluation
        
        Args:
            reference: Reference prompt text.
            hypothesis: ASR transcription text.
            
        Returns:
            Result dictionary
        """
        # 1. Completeness
        completeness = self.calculate_completeness(reference, hypothesis)
        
        # 2. Accuracy
        accuracy_metrics = self.calculate_accuracy(reference, hypothesis)
        
        # 3. No-hallucination score
        hallucination_score = self.calculate_hallucination(reference, hypothesis)
        
        # 4. Weighted overall score
        overall_score = (
            completeness * 0.4 +           # Completeness weight 40%
            accuracy_metrics['accuracy'] * 0.4 +  # Accuracy weight 40%
            hallucination_score * 0.2      # No-hallucination weight 20%
        )
        
        return {
            'completeness': completeness,
            'accuracy': accuracy_metrics['accuracy'],
            'edit_distance_similarity': accuracy_metrics['edit_distance_similarity'],
            'word_overlap': accuracy_metrics['word_overlap'],
            'char_overlap': accuracy_metrics['char_overlap'],
            'hallucination_score': hallucination_score,
            'overall_content_score': overall_score,
            'transcription': hypothesis,
            'reference': reference
        }
    
    def evaluate_video(self, video_path: str, prompt_text: str) -> Dict:
        """
        Evaluate speech content for a single video
        
        Args:
            video_path: Video file path
            prompt_text: Prompt text
            
        Returns:
            Evaluation result
        """
        # Extract audio
        temp_audio = tempfile.mktemp(suffix=".wav")
        
        try:
            if not extract_audio_from_video(video_path, temp_audio):
                return {
                    'error': 'Audio extraction failed',
                    'completeness': 0.0,
                    'accuracy': 0.0,
                    'hallucination_score': 0.0,
                    'overall_content_score': 0.0
                }
            
            # Run ASR transcription.
            transcription = self.transcribe_audio(temp_audio)
            
            # Evaluate content
            result = self.evaluate_content(prompt_text, transcription)
            
            return result
            
        finally:
            # Clean temporary files
            if os.path.exists(temp_audio):
                os.remove(temp_audio)


def load_prompts(prompt_file: str = None) -> Dict[str, str]:
    """
    Load prompt file
    
    Args:
        prompt_file: Prompt file path (JSON)
        
    Returns:
        Mapping from video_id to prompt_text.
    """
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Return empty dict by default
    print("Warning: prompt file not provided; using empty prompts")
    return {}


def evaluate_videos_content(
    video_dir: str,
    prompt_file: str,
    output_csv: str,
    asr_model: str = "openai/whisper-large-v3",
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Batch-evaluate speech content accuracy
    
    Args:
        video_dir: Video directory
        prompt_file: Prompt JSON file
        output_csv: Output CSV path
        asr_model: ASR model
        device: Runtime device
        
    Returns:
        DataFrame with evaluation results.
    """
    print("=" * 60)
    print("Speech Content Accuracy Evaluation")
    print("=" * 60)
    print(f"Video directory: {video_dir}")
    print(f"Prompt file: {prompt_file}")
    print(f"Output file: {output_csv}")
    print(f"ASR model: {asr_model}")
    print(f"Runtime device: {device}")
    print()
    
    # Load prompts.
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts\n")
    
    # Initialize evaluator
    evaluator = SpeechContentEvaluator(asr_model, device)
    
    # Find video files
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(video_files)} video files\n")
    
    if not video_files:
        print("Error: no videos found")
        return pd.DataFrame()
    
    results_list = []
    
    print("Start evaluation...")
    for video_path in tqdm(video_files, desc="Evaluating"):
        video_name = video_path.name
        # Video filename may be hash-only or hash_suffix format
        parts = video_path.stem.split('_')
        if len(parts) >= 2:
            video_id = parts[0] + '_' + parts[1]
        else:
            video_id = video_path.stem
        
        # Match by full filename first, then video_id, then stem
        prompt_text = prompts.get(video_name,
                      prompts.get(video_id,
                      prompts.get(video_path.stem, "")))
        
        if not prompt_text:
            print(f"Warning: video {video_name} has no matching prompt")
        
        # Evaluate one video.
        result = evaluator.evaluate_video(str(video_path), prompt_text)
        
        results_list.append({
            'video_id': video_id,
            'video_name': video_name,
            'prompt_text': prompt_text,
            'transcription': result.get('transcription', ''),
            'completeness': result['completeness'],
            'accuracy': result['accuracy'],
            'edit_distance_similarity': result.get('edit_distance_similarity', 0),
            'word_overlap': result.get('word_overlap', 0),
            'char_overlap': result.get('char_overlap', 0),
            'hallucination_score': result['hallucination_score'],
            'overall_content_score': result['overall_content_score']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Save results
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    print(f"Total videos: {len(df)}")
    print()
    
    # Primary metrics
    print("=" * 60)
    print("【Primary metrics】")
    print("=" * 60)
    print(f"1. Completeness (Completeness):        {df['completeness'].mean():.2f} ± {df['completeness'].std():.2f}")
    print(f"   - Max: {df['completeness'].max():.2f}  Min: {df['completeness'].min():.2f}")
    print(f"   - Median: {df['completeness'].median():.2f}")
    print()
    
    print(f"2. Accuracy (Accuracy):            {df['accuracy'].mean():.2f} ± {df['accuracy'].std():.2f}")
    print(f"   - Max: {df['accuracy'].max():.2f}  Min: {df['accuracy'].min():.2f}")
    print(f"   - Median: {df['accuracy'].median():.2f}")
    print()
    
    print(f"3. No-hallucination score (No Hallucination): {df['hallucination_score'].mean():.2f} ± {df['hallucination_score'].std():.2f}")
    print(f"   - Max: {df['hallucination_score'].max():.2f}  Min: {df['hallucination_score'].min():.2f}")
    print(f"   - Median: {df['hallucination_score'].median():.2f}")
    print()
    
    print(f"4. Overall score (Overall):           {df['overall_content_score'].mean():.2f} ± {df['overall_content_score'].std():.2f}")
    print(f"   - Max: {df['overall_content_score'].max():.2f}  Min: {df['overall_content_score'].min():.2f}")
    print(f"   - Median: {df['overall_content_score'].median():.2f}")
    print()
    
    # Sub-metric statistics.
    print("=" * 60)
    print("【Detailed sub-metrics】")
    print("=" * 60)
    print(f"Edit-distance similarity (Edit Distance):  {df['edit_distance_similarity'].mean():.2f} ± {df['edit_distance_similarity'].std():.2f}")
    print(f"   - Max: {df['edit_distance_similarity'].max():.2f}  Min: {df['edit_distance_similarity'].min():.2f}")
    print()
    
    print(f"Word-level overlap (Word Overlap):       {df['word_overlap'].mean():.2f} ± {df['word_overlap'].std():.2f}")
    print(f"   - Max: {df['word_overlap'].max():.2f}  Min: {df['word_overlap'].min():.2f}")
    print()
    
    print(f"Character-level overlap (Char Overlap):     {df['char_overlap'].mean():.2f} ± {df['char_overlap'].std():.2f}")
    print(f"   - Max: {df['char_overlap'].max():.2f}  Min: {df['char_overlap'].min():.2f}")
    print()
    
    # Score distribution
    print("=" * 60)
    print("【Score distribution】")
    print("=" * 60)
    
    def print_score_distribution(scores, metric_name):
        excellent = (scores >= 90).sum()
        good = ((scores >= 80) & (scores < 90)).sum()
        fair = ((scores >= 70) & (scores < 80)).sum()
        poor = ((scores >= 60) & (scores < 70)).sum()
        bad = (scores < 60).sum()
        total = len(scores)
        
        print(f"{metric_name}:")
        print(f"  Excellent (>=90): {excellent:2d} ({excellent/total*100:5.1f}%)")
        print(f"  Good (80-89): {good:2d} ({good/total*100:5.1f}%)")
        print(f"  Fair (70-79): {fair:2d} ({fair/total*100:5.1f}%)")
        print(f"  Pass (60-69): {poor:2d} ({poor/total*100:5.1f}%)")
        print(f"  Fail (<60): {bad:2d} ({bad/total*100:5.1f}%)")
        print()
    
    print_score_distribution(df['completeness'], "Completeness distribution")
    print_score_distribution(df['accuracy'], "Accuracy distribution")
    print_score_distribution(df['hallucination_score'], "No-hallucination score distribution")
    print_score_distribution(df['overall_content_score'], "Overall score distribution")
    
    # Best and worst videos.
    print("=" * 60)
    print("【Top 5 best videos】(Overall score)")
    print("=" * 60)
    top5 = df.nlargest(5, 'overall_content_score')
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['video_id']}: {row['overall_content_score']:.2f}")
        print(f"   Completeness:{row['completeness']:.1f} Accuracy:{row['accuracy']:.1f} No-hallucination:{row['hallucination_score']:.1f}")
    print()
    
    print("=" * 60)
    print("【Top 5 worst videos】(Overall score)")
    print("=" * 60)
    bottom5 = df.nsmallest(5, 'overall_content_score')
    for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
        print(f"{i}. {row['video_id']}: {row['overall_content_score']:.2f}")
        print(f"   Completeness:{row['completeness']:.1f} Accuracy:{row['accuracy']:.1f} No-hallucination:{row['hallucination_score']:.1f}")
    
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Multi-dataset batch evaluation entry
# ──────────────────────────────────────────────────────────────────────────────

def discover_dataset_jsons(dataset_root: str) -> List[str]:
    """Find all JSON files under dataset root for utterance mapping."""
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*.json"))


def build_utterance_map(json_path: str) -> Dict[str, str]:
    """
    Build hash -> utterance mapping from dataset JSON.
    JSON format: list with items containing 'video_file' (e.g. 'abc123.mp4') and 'utterance'.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping = {}
    for item in data:
        utterance = str(item.get('utterance', '')).strip()
        if not utterance:
            continue

        # Support multiple open-source dataset schemas.
        # Priority: video_file -> id -> video_id -> hash
        if item.get('video_file'):
            sample_id = Path(str(item['video_file'])).stem
        else:
            sample_id = str(
                item.get('id')
                or item.get('video_id')
                or item.get('hash')
                or ''
            ).strip()

        if sample_id:
            mapping[sample_id] = utterance
    return mapping


def build_merged_utterance_map(json_paths: List[str]) -> Dict[str, str]:
    """Merge hash->utterance mappings from multiple dataset JSON files."""
    merged: Dict[str, str] = {}
    for p in json_paths:
        try:
            merged.update(build_utterance_map(p))
        except Exception as e:
            print(f"Warning: skip invalid dataset file {p}: {e}")
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speech content accuracy for AVBench videos.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--dataset-root", default=str(default_dataset_root()), help="Root directory of text/prompt datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--device", default="cuda", help="ASR inference device.")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3", help="ASR model name/path for transformers backend.")
    return parser.parse_args()


def load_done_hashes(video_dir: str) -> set:
    """
    Read video_dir/generate_state.json and return hashes with status==done.
    Return None when file does not exist (no filtering).
    """
    state_path = os.path.join(video_dir, 'generate_state.json')
    if not os.path.exists(state_path):
        return None
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    done = set()
    skipped = 0
    for video_file, info in state.items():
        if info.get('status') == 'done':
            done.add(Path(video_file).stem)   # hash without extension
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} non-done videos (failed/submitted)")
    return done


def evaluate_videos_batch(
    video_dir: str,
    utterance_map: Dict[str, str],
    output_csv: str,
    evaluator: 'SpeechContentEvaluator',
) -> pd.DataFrame:
    """
    Batch-evaluate speech content accuracy for one video directory.

    Expected video filename: {hash}_{model_suffix}.mp4 or {hash}.mp4
    Ground-truth text is looked up by hash in utterance_map.
    Videos with status != 'done' in generate_state.json are skipped.
    """
    # Read successful-generation set
    done_hashes = load_done_hashes(video_dir)  # None means no state file (no filtering)

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        print(f"  Skipping: no mp4 files found in directory ({video_dir})")
        return pd.DataFrame()

    results_list = []
    no_ref_count = 0
    skipped_not_done = 0

    for video_path in tqdm(video_files, desc=f"  Evaluating {Path(video_dir).name}"):
        stem    = video_path.stem                        # "abc123_kling" or "abc123"
        hash_id = stem.split('_')[0]                     # Use first segment as hash

        # Skip failed-generation videos
        if done_hashes is not None and hash_id not in done_hashes:
            skipped_not_done += 1
            continue

        utterance = utterance_map.get(hash_id, "")
        if not utterance:
            no_ref_count += 1

        result = evaluator.evaluate_video(str(video_path), utterance)

        results_list.append({
            'video_name':              video_path.name,
            'hash_id':                 hash_id,
            'reference_utterance':     utterance,
            'transcription':           result.get('transcription', ''),
            'completeness':            result['completeness'],
            'accuracy':                result['accuracy'],
            'edit_distance_similarity':result.get('edit_distance_similarity', 0),
            'word_overlap':            result.get('word_overlap', 0),
            'char_overlap':            result.get('char_overlap', 0),
            'hallucination_score':     result['hallucination_score'],
            'overall_content_score':   result['overall_content_score'],
        })

    if skipped_not_done:
        print(f"  Skipped {skipped_not_done} failed/not-done videos")
    if no_ref_count:
        print(f"  Warning: {no_ref_count} videos are missing corresponding utterance")

    df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return df


def main():
    args = parse_args()
    device = args.device
    results_dir = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
    dataset_jsons = discover_dataset_jsons(args.dataset_root)
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"No evaluable video directories found under: {args.video_root}")
        return
    if not dataset_jsons:
        print(f"No usable text dataset JSON files found under: {args.dataset_root}")
        return

    utterance_map = build_merged_utterance_map(dataset_jsons)
    print(f"Merged utterance map size: {len(utterance_map)}")

    # Initialize ASR evaluator once and reuse across all datasets.
    evaluator = SpeechContentEvaluator(asr_model=args.asr_model, device=device)

    summary_rows = []

    for label, video_dir in video_dirs:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {label}  ({video_dir})")
        print(f"{'#'*70}")

        if not os.path.exists(video_dir):
            print("  Skipping: directory does not exist")
            continue

        output_csv = os.path.join(results_dir, f"speech_content_{label}.csv")
        df = evaluate_videos_batch(video_dir, utterance_map, output_csv, evaluator)

        if df.empty:
            continue

        mean_overall    = float(df['overall_content_score'].mean())
        mean_complete   = float(df['completeness'].mean())
        mean_accuracy   = float(df['accuracy'].mean())
        mean_halluc     = float(df['hallucination_score'].mean())

        print(f"\n  Results -> {output_csv}")
        print(f"  Overall: {mean_overall:.2f}  Completeness: {mean_complete:.2f}  "
              f"Accuracy: {mean_accuracy:.2f}  No-hallucination: {mean_halluc:.2f}")

        summary_rows.append({
            "dataset":               label,
            "n_videos":              len(df),
            "mean_overall":          round(mean_overall,  4),
            "mean_completeness":     round(mean_complete, 4),
            "mean_accuracy":         round(mean_accuracy, 4),
            "mean_hallucination":    round(mean_halluc,   4),
        })

    # Write summary
    summary_path = os.path.join(results_dir, "speech_content_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*70}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
