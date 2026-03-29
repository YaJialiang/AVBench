"""Evaluate speech content fidelity (completeness/accuracy/hallucination) for AVBench videos."""

import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
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
    """从视频中提取音频为16kHz单声道WAV"""
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
        print(f"提取音频失败 {video_path}: {e}")
        return False


class SpeechContentEvaluator:
    """语音内容评估器"""
    
    def __init__(self, asr_model="openai/whisper-large-v3", device="cuda"):
        """
        初始化评估器
        
        Args:
            asr_model: ASR模型路径或名称
            device: 运行设备
        """
        self.device = device
        print("正在加载ASR模型...")
        
        # 导入whisper
        try:
            import whisper
            self.whisper_model = whisper.load_model("large-v3", device=device)
            self.use_transformers = False
            print("✓ 使用 OpenAI Whisper 模型\n")
        except:
            # 如果没有whisper，使用transformers版本
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
                # device_map="auto" 已由模型接管，不能再传 device
            )
            self.use_transformers = True
            print(f"✓ 使用 Transformers Whisper 模型\n")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        使用ASR识别音频内容
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别出的文本
        """
        try:
            if self.use_transformers:
                # language=None 自动检测语言（多语言数据集）
                result = self.asr_pipeline(audio_path, generate_kwargs={"language": None})
                return result['text'].strip()
            else:
                # language=None 自动检测（支持中/英/日/韩等多语言）
                result = self.whisper_model.transcribe(audio_path, language=None)
                return result['text'].strip()
        except Exception as e:
            print(f"ASR识别失败: {e}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """
        文本标准化：去除标点、统一空格
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        # 转小写
        text = text.lower()
        
        # 去除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        
        # 统一空格
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（简单实现：去除停用词后的词汇）
        
        Args:
            text: 文本
            
        Returns:
            关键词列表
        """
        # 简单的中文停用词
        stopwords = set(['的', '了', '在', '是', '我', '你', '他', '她', '它', 
                        '们', '这', '那', '有', '个', '和', '与', '或', '但',
                        '啊', '呀', '吗', '呢', '吧', '嘛'])
        
        # 分词（简单按字符分割）
        words = list(text)
        
        # 过滤停用词
        keywords = [w for w in words if w not in stopwords and w.strip()]
        
        return keywords
    
    def calculate_completeness(self, reference: str, hypothesis: str) -> float:
        """
        计算内容完整度：参考文本中的关键词在识别文本中出现的比例
        
        Args:
            reference: 参考文本（提示词）
            hypothesis: 识别文本（ASR输出）
            
        Returns:
            完整度分数 (0-100)
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # 提取关键词
        ref_keywords = self.extract_keywords(ref_norm)
        
        if not ref_keywords:
            return 100.0
        
        # 计算有多少关键词被覆盖
        covered = sum(1 for kw in ref_keywords if kw in hyp_norm)
        completeness = (covered / len(ref_keywords)) * 100
        
        return completeness
    
    def calculate_accuracy(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        计算内容准确度：多种相似度指标
        
        Args:
            reference: 参考文本（提示词）
            hypothesis: 识别文本（ASR输出）
            
        Returns:
            包含多种准确度指标的字典
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # 1. 字符级编辑距离相似度
        edit_distance_ratio = difflib.SequenceMatcher(None, ref_norm, hyp_norm).ratio()
        
        # 2. 词级重叠度
        ref_words = set(ref_norm.split())
        hyp_words = set(hyp_norm.split())
        
        if not ref_words:
            word_overlap = 1.0
        else:
            word_overlap = len(ref_words & hyp_words) / len(ref_words)
        
        # 3. 字符级重叠度
        ref_chars = set(ref_norm)
        hyp_chars = set(hyp_norm)
        
        if not ref_chars:
            char_overlap = 1.0
        else:
            char_overlap = len(ref_chars & hyp_chars) / len(ref_chars)
        
        # 综合准确度（加权平均）
        accuracy = (edit_distance_ratio * 0.5 + word_overlap * 0.3 + char_overlap * 0.2) * 100
        
        return {
            'edit_distance_similarity': edit_distance_ratio * 100,
            'word_overlap': word_overlap * 100,
            'char_overlap': char_overlap * 100,
            'accuracy': accuracy
        }
    
    def calculate_hallucination(self, reference: str, hypothesis: str) -> float:
        """
        计算幻觉分数：识别文本中有多少内容是参考文本没有的
        
        Args:
            reference: 参考文本（提示词）
            hypothesis: 识别文本（ASR输出）
            
        Returns:
            幻觉分数 (0-100)，100表示没有多余内容
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        # 提取识别文本的关键词
        hyp_keywords = self.extract_keywords(hyp_norm)
        
        if not hyp_keywords:
            return 100.0
        
        # 计算有多少词是参考文本中没有的
        hallucinated = sum(1 for kw in hyp_keywords if kw not in ref_norm)
        hallucination_rate = hallucinated / len(hyp_keywords)
        
        # 转换为分数（越少越好）
        hallucination_score = (1 - hallucination_rate) * 100
        
        return hallucination_score
    
    def evaluate_content(self, reference: str, hypothesis: str) -> Dict:
        """
        综合评估语音内容
        
        Args:
            reference: 参考文本（提示词）
            hypothesis: 识别文本（ASR输出）
            
        Returns:
            评估结果字典
        """
        # 1. 完整度
        completeness = self.calculate_completeness(reference, hypothesis)
        
        # 2. 准确度
        accuracy_metrics = self.calculate_accuracy(reference, hypothesis)
        
        # 3. 幻觉分数
        hallucination_score = self.calculate_hallucination(reference, hypothesis)
        
        # 4. 综合分数（加权平均）
        overall_score = (
            completeness * 0.4 +           # 完整度权重40%
            accuracy_metrics['accuracy'] * 0.4 +  # 准确度权重40%
            hallucination_score * 0.2      # 无幻觉权重20%
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
        评估单个视频的语音内容
        
        Args:
            video_path: 视频文件路径
            prompt_text: 提示词文本
            
        Returns:
            评估结果
        """
        # 提取音频
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
            
            # ASR识别
            transcription = self.transcribe_audio(temp_audio)
            
            # 评估内容
            result = self.evaluate_content(prompt_text, transcription)
            
            return result
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)


def load_prompts(prompt_file: str = None) -> Dict[str, str]:
    """
    加载提示词文件
    
    Args:
        prompt_file: 提示词文件路径（JSON格式）
        
    Returns:
        video_id -> prompt_text 的字典
    """
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认返回空字典
    print("警告：未提供提示词文件，将使用空提示词")
    return {}


def evaluate_videos_content(
    video_dir: str,
    prompt_file: str,
    output_csv: str,
    asr_model: str = "openai/whisper-large-v3",
    device: str = "cuda"
) -> pd.DataFrame:
    """
    批量评估视频的语音内容准确性
    
    Args:
        video_dir: 视频目录
        prompt_file: 提示词JSON文件
        output_csv: 输出CSV路径
        asr_model: ASR模型
        device: 运行设备
        
    Returns:
        包含评估结果的DataFrame
    """
    print("=" * 60)
    print("语音内容准确性评测")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"提示词文件: {prompt_file}")
    print(f"输出文件: {output_csv}")
    print(f"ASR模型: {asr_model}")
    print(f"运行设备: {device}")
    print()
    
    # 加载提示词
    prompts = load_prompts(prompt_file)
    print(f"加载了 {len(prompts)} 个提示词\n")
    
    # 初始化评估器
    evaluator = SpeechContentEvaluator(asr_model, device)
    
    # 查找视频文件
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    if not video_files:
        print("错误：未找到视频文件！")
        return pd.DataFrame()
    
    results_list = []
    
    print("开始评估...")
    for video_path in tqdm(video_files, desc="评估进度"):
        video_name = video_path.name
        # 视频文件名可能是 hash（无下划线），也可能是 video_000001_xxx 格式
        parts = video_path.stem.split('_')
        if len(parts) >= 2:
            video_id = parts[0] + '_' + parts[1]
        else:
            video_id = video_path.stem
        
        # 优先用完整文件名（含 .mp4）匹配，其次 video_id，最后 stem
        prompt_text = prompts.get(video_name,
                      prompts.get(video_id,
                      prompts.get(video_path.stem, "")))
        
        if not prompt_text:
            print(f"警告：视频 {video_name} 没有对应的提示词")
        
        # 评估
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
    
    # 创建DataFrame
    df = pd.DataFrame(results_list)
    
    # 保存结果
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"总视频数: {len(df)}")
    print()
    
    # 主要指标统计
    print("=" * 60)
    print("【主要指标统计】")
    print("=" * 60)
    print(f"1. 完整度 (Completeness):        {df['completeness'].mean():.2f} ± {df['completeness'].std():.2f}")
    print(f"   - 最高: {df['completeness'].max():.2f}  最低: {df['completeness'].min():.2f}")
    print(f"   - 中位数: {df['completeness'].median():.2f}")
    print()
    
    print(f"2. 准确度 (Accuracy):            {df['accuracy'].mean():.2f} ± {df['accuracy'].std():.2f}")
    print(f"   - 最高: {df['accuracy'].max():.2f}  最低: {df['accuracy'].min():.2f}")
    print(f"   - 中位数: {df['accuracy'].median():.2f}")
    print()
    
    print(f"3. 无幻觉分数 (No Hallucination): {df['hallucination_score'].mean():.2f} ± {df['hallucination_score'].std():.2f}")
    print(f"   - 最高: {df['hallucination_score'].max():.2f}  最低: {df['hallucination_score'].min():.2f}")
    print(f"   - 中位数: {df['hallucination_score'].median():.2f}")
    print()
    
    print(f"4. 综合分数 (Overall):           {df['overall_content_score'].mean():.2f} ± {df['overall_content_score'].std():.2f}")
    print(f"   - 最高: {df['overall_content_score'].max():.2f}  最低: {df['overall_content_score'].min():.2f}")
    print(f"   - 中位数: {df['overall_content_score'].median():.2f}")
    print()
    
    # 子指标统计
    print("=" * 60)
    print("【子指标详细统计】")
    print("=" * 60)
    print(f"编辑距离相似度 (Edit Distance):  {df['edit_distance_similarity'].mean():.2f} ± {df['edit_distance_similarity'].std():.2f}")
    print(f"   - 最高: {df['edit_distance_similarity'].max():.2f}  最低: {df['edit_distance_similarity'].min():.2f}")
    print()
    
    print(f"词级重叠度 (Word Overlap):       {df['word_overlap'].mean():.2f} ± {df['word_overlap'].std():.2f}")
    print(f"   - 最高: {df['word_overlap'].max():.2f}  最低: {df['word_overlap'].min():.2f}")
    print()
    
    print(f"字符级重叠度 (Char Overlap):     {df['char_overlap'].mean():.2f} ± {df['char_overlap'].std():.2f}")
    print(f"   - 最高: {df['char_overlap'].max():.2f}  最低: {df['char_overlap'].min():.2f}")
    print()
    
    # 分数分布统计
    print("=" * 60)
    print("【分数分布统计】")
    print("=" * 60)
    
    def print_score_distribution(scores, metric_name):
        excellent = (scores >= 90).sum()
        good = ((scores >= 80) & (scores < 90)).sum()
        fair = ((scores >= 70) & (scores < 80)).sum()
        poor = ((scores >= 60) & (scores < 70)).sum()
        bad = (scores < 60).sum()
        total = len(scores)
        
        print(f"{metric_name}:")
        print(f"  优秀 (≥90分): {excellent:2d} ({excellent/total*100:5.1f}%)")
        print(f"  良好 (80-89): {good:2d} ({good/total*100:5.1f}%)")
        print(f"  中等 (70-79): {fair:2d} ({fair/total*100:5.1f}%)")
        print(f"  及格 (60-69): {poor:2d} ({poor/total*100:5.1f}%)")
        print(f"  不及格 (<60): {bad:2d} ({bad/total*100:5.1f}%)")
        print()
    
    print_score_distribution(df['completeness'], "完整度分布")
    print_score_distribution(df['accuracy'], "准确度分布")
    print_score_distribution(df['hallucination_score'], "无幻觉分数分布")
    print_score_distribution(df['overall_content_score'], "综合分数分布")
    
    # 最佳和最差视频
    print("=" * 60)
    print("【最佳视频 TOP 5】(综合分数)")
    print("=" * 60)
    top5 = df.nlargest(5, 'overall_content_score')
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['video_id']}: {row['overall_content_score']:.2f}")
        print(f"   完整度:{row['completeness']:.1f} 准确度:{row['accuracy']:.1f} 无幻觉:{row['hallucination_score']:.1f}")
    print()
    
    print("=" * 60)
    print("【最差视频 TOP 5】(综合分数)")
    print("=" * 60)
    bottom5 = df.nsmallest(5, 'overall_content_score')
    for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
        print(f"{i}. {row['video_id']}: {row['overall_content_score']:.2f}")
        print(f"   完整度:{row['completeness']:.1f} 准确度:{row['accuracy']:.1f} 无幻觉:{row['hallucination_score']:.1f}")
    
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 多数据集批量评估入口
# ──────────────────────────────────────────────────────────────────────────────

def discover_dataset_jsons(dataset_root: str) -> List[str]:
    """Find all JSON files under dataset root for utterance mapping."""
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*.json"))


def build_utterance_map(json_path: str) -> Dict[str, str]:
    """
    从数据集 JSON 构建 hash → utterance 映射。
    JSON 格式为列表，每项含 'video_file'（如 'abc123.mp4'）和 'utterance'。
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
            print(f"警告：跳过无效数据集文件 {p}: {e}")
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
    读取 video_dir/generate_state.json，返回 status=='done' 的 hash 集合。
    若文件不存在则返回 None（表示不过滤）。
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
            done.add(Path(video_file).stem)   # hash（不含扩展名）
        else:
            skipped += 1
    if skipped:
        print(f"  跳过 {skipped} 个非 done 视频（failed/submitted）")
    return done


def evaluate_videos_batch(
    video_dir: str,
    utterance_map: Dict[str, str],
    output_csv: str,
    evaluator: 'SpeechContentEvaluator',
) -> pd.DataFrame:
    """
    对单个视频目录批量评估语音内容准确性。

    视频文件名格式：{hash}_{model_suffix}.mp4 或 {hash}.mp4
    通过 hash 在 utterance_map 中查找 ground-truth 文本。
    跳过 generate_state.json 中 status != 'done' 的视频。
    """
    # 读取生成成功集合
    done_hashes = load_done_hashes(video_dir)  # None 表示无 state 文件，不过滤

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        print(f"  跳过：目录无 mp4 文件 ({video_dir})")
        return pd.DataFrame()

    results_list = []
    no_ref_count = 0
    skipped_not_done = 0

    for video_path in tqdm(video_files, desc=f"  评估 {Path(video_dir).name}"):
        stem    = video_path.stem                        # "abc123_kling" 或 "abc123"
        hash_id = stem.split('_')[0]                     # 取第一段作为 hash

        # 跳过生成失败的视频
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
        print(f"  已跳过 {skipped_not_done} 个生成失败视频")
    if no_ref_count:
        print(f"  警告：{no_ref_count} 个视频未找到对应 utterance")

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
        print(f"未发现可评估视频目录: {args.video_root}")
        return
    if not dataset_jsons:
        print(f"未发现可用文本数据集 JSON: {args.dataset_root}")
        return

    utterance_map = build_merged_utterance_map(dataset_jsons)
    print(f"Merged utterance map size: {len(utterance_map)}")

    # 初始化 ASR 评估器（只加载一次）
    evaluator = SpeechContentEvaluator(asr_model=args.asr_model, device=device)

    summary_rows = []

    for label, video_dir in video_dirs:
        print(f"\n{'#'*70}")
        print(f"# 数据集: {label}  ({video_dir})")
        print(f"{'#'*70}")

        if not os.path.exists(video_dir):
            print("  跳过：目录不存在")
            continue

        output_csv = os.path.join(results_dir, f"speech_content_{label}.csv")
        df = evaluate_videos_batch(video_dir, utterance_map, output_csv, evaluator)

        if df.empty:
            continue

        mean_overall    = float(df['overall_content_score'].mean())
        mean_complete   = float(df['completeness'].mean())
        mean_accuracy   = float(df['accuracy'].mean())
        mean_halluc     = float(df['hallucination_score'].mean())

        print(f"\n  结果 → {output_csv}")
        print(f"  综合分: {mean_overall:.2f}  完整度: {mean_complete:.2f}  "
              f"准确度: {mean_accuracy:.2f}  无幻觉: {mean_halluc:.2f}")

        summary_rows.append({
            "dataset":               label,
            "n_videos":              len(df),
            "mean_overall":          round(mean_overall,  4),
            "mean_completeness":     round(mean_complete, 4),
            "mean_accuracy":         round(mean_accuracy, 4),
            "mean_hallucination":    round(mean_halluc,   4),
        })

    # 写入汇总
    summary_path = os.path.join(results_dir, "speech_content_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*70}")
    print(f"汇总结果已写入: {summary_path}")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
