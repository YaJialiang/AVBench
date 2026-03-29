#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate AVBench video audio quality with Audiobox Aesthetics metrics."""

import sys
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
    """查找目录下的所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    return sorted([str(f) for f in video_files])


def extract_video_id(video_path: str) -> str:
    """从视频文件名中提取ID"""
    filename = Path(video_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return filename


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        
    Returns:
        提取的音频文件路径
    """
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    
    # 使用 ffmpeg 提取音频
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # 不处理视频
        '-acodec', 'pcm_s16le',  # 使用 PCM 编码
        '-ar', '16000',  # 采样率 16kHz
        '-ac', '1',  # 单声道
        '-y',  # 覆盖已存在的文件
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频提取失败: {e.stderr}")


def evaluate_audio_aesthetics(audio_paths: List[str], 
                              checkpoint_path: str = None,
                              batch_size: int = 16) -> List[Dict[str, float]]:
    """
    使用 Audiobox Aesthetics 评估音频质量
    
    Args:
        audio_paths: 音频文件路径列表
        checkpoint_path: 模型检查点路径
        batch_size: 批处理大小
        
    Returns:
        评估结果列表
    """
    # 创建输入文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        input_file = f.name
        for audio_path in audio_paths:
            f.write(json.dumps({"path": audio_path}) + '\n')
    
    try:
        # 构建命令
        cmd = ['audio-aes', input_file, '--batch-size', str(batch_size)]
        
        if checkpoint_path:
            cmd.extend(['--ckpt', checkpoint_path])
        
        # 运行评估
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        # 解析输出
        results = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                results.append(json.loads(line))
        
        return results
        
    finally:
        # 清理临时文件
        if os.path.exists(input_file):
            os.remove(input_file)


def evaluate_videos(video_dir: str,
                    output_path: str,
                    checkpoint_path: str = None,
                    batch_size: int = 16,
                    keep_audio: bool = False):
    """
    评估所有视频的音频美学质量
    
    Args:
        video_dir: 视频目录
        output_path: 输出CSV文件路径
        checkpoint_path: Audiobox模型检查点路径
        batch_size: 批处理大小
        keep_audio: 是否保留提取的音频文件
    """
    print("=" * 80)
    print("Audiobox Aesthetics 音频质量评估")
    print("=" * 80)
    
    # 查找所有视频文件
    video_files = find_video_files(video_dir)
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        print(f"错误: 在 {video_dir} 中未找到视频文件")
        return
    
    # 创建临时目录存储音频
    audio_dir = tempfile.mkdtemp(prefix='va_bench_audio_')
    print(f"\n音频提取目录: {audio_dir}")
    
    try:
        # 第一步：从视频提取音频
        print("\n步骤 1/2: 从视频提取音频...")
        audio_paths = []
        video_audio_map = {}
        
        for video_path in tqdm(video_files, desc="提取音频"):
            try:
                audio_path = extract_audio_from_video(video_path, audio_dir)
                audio_paths.append(audio_path)
                video_audio_map[audio_path] = video_path
            except Exception as e:
                print(f"\n警告: 提取 {Path(video_path).name} 音频失败: {str(e)}")
        
        if not audio_paths:
            print("错误: 没有成功提取任何音频")
            return
        
        print(f"成功提取 {len(audio_paths)} 个音频文件")
        
        # 第二步：评估音频美学质量
        print(f"\n步骤 2/2: 评估音频美学质量...")
        print(f"使用 Audiobox Aesthetics 模型...")
        print(f"评估指标: CE(内容享受度), CU(内容有用性), PC(制作复杂度), PQ(制作质量)")
        
        aesthetics_results = evaluate_audio_aesthetics(
            audio_paths, 
            checkpoint_path=checkpoint_path,
            batch_size=batch_size
        )
        
        # 整合结果
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
                # 论文公式: S = (CE + CU + PQ - PC) / 4
                'avg_score': (
                    aesthetics.get('CE', 0)
                    + aesthetics.get('CU', 0)
                    + aesthetics.get('PQ', 0)
                    - aesthetics.get('PC', 0)
                ) / 4
            })
        
        # 保存结果
        df = pd.DataFrame(results)
        df = df.sort_values('video_id')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n评估完成，结果已保存到: {output_path}")
        return df
        
    finally:
        if not keep_audio:
            print(f"\n清理临时音频文件...")
            import shutil
            if os.path.exists(audio_dir):
                shutil.rmtree(audio_dir)
        else:
            print(f"\n保留音频文件于: {audio_dir}")


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
        print(f"警告: 未找到检查点 {checkpoint_path}，将使用默认模型")
        checkpoint_path = None

    # 检查 audio-aes 命令是否可用
    try:
        subprocess.run(['audio-aes', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到 audio-aes 命令\n请安装: pip install audiobox-aesthetics")
        return

    batch_size  = args.batch_size
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"未发现可评估视频目录: {args.video_root}")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*80}")
        print(f"# 数据集: {label}  ({video_dir})")
        print(f"{'#'*80}")
        if not os.path.exists(video_dir):
            print("跳过：目录不存在")
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
    print(f"汇总结果已写入: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
