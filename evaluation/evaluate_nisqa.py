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


def extract_audio_from_video(video_path: str, output_dir: str = None) -> str:
    """
    从视频中提取音频并保存为 WAV 文件
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（如果为 None，使用临时目录）
        
    Returns:
        提取的音频文件路径
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    
    # 使用 ffmpeg 提取音频
    # NISQA 推荐使用 16kHz 或 48kHz 采样率
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
    """
    评估所有视频的语音质量
    
    Args:
        video_dir: 视频目录
        output_path: 输出CSV文件路径
        model_path: NISQA模型权重路径
        use_tts_model: 是否使用 TTS 模型（评估自然度）
    """
    print("=" * 80)
    print("NISQA v2.0 语音质量综合评估")
    print("=" * 80)
    
    if use_tts_model:
        print("\n模型类型: NISQA-TTS (合成语音自然度评估)")
        print("评估指标: Naturalness (自然度)")
    else:
        print("\n模型类型: NISQA v2.0 (传输语音质量评估)")
        print("评估指标: MOS, Noisiness, Coloration, Discontinuity, Loudness")
    
    # 查找所有视频文件
    video_files = find_video_files(video_dir)
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        print(f"错误: 在 {video_dir} 中未找到视频文件")
        return
    
    # 创建临时目录用于音频提取
    audio_dir = tempfile.mkdtemp(prefix='va_bench_nisqa_')
    print(f"\n音频提取目录: {audio_dir}")
    
    try:
        # 第一步：提取音频
        print(f"\n步骤 1/3: 从视频提取音频...")
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
        
        # 第二步：创建 CSV 文件用于 NISQA 预测
        print(f"\n步骤 2/3: 准备 NISQA 输入...")
        csv_input_path = os.path.join(audio_dir, 'nisqa_input.csv')
        df_input = pd.DataFrame({
            'filepath': audio_paths
        })
        df_input.to_csv(csv_input_path, index=False)
        print(f"输入文件已创建: {csv_input_path}")
        
        # 第三步：运行 NISQA 预测
        print(f"\n步骤 3/3: 运行 NISQA 评估...")
        print(f"这可能需要几分钟时间...")
        
        # 配置 NISQA 参数
        args = {
            'mode': 'predict_csv',
            'pretrained_model': model_path,
            'csv_file': csv_input_path,
            'csv_deg': 'filepath',
            'data_dir': '',
            'output_dir': audio_dir,
            'num_workers': 0,
            'bs': 4,  # 批处理大小
            'tr_bs_val': 4,
            'tr_num_workers': 0,
            'ms_channel': None
        }
        
        # 初始化并运行 NISQA 模型
        nisqaModel = load_nisqa_model_class(nisqa_root)
        nisqa = nisqaModel(args)
        nisqa.predict()
        
        # 读取 NISQA 输出结果
        nisqa_output_path = os.path.join(audio_dir, 'NISQA_results.csv')
        if not os.path.exists(nisqa_output_path):
            print(f"错误: NISQA 输出文件不存在: {nisqa_output_path}")
            return
        
        df_nisqa = pd.read_csv(nisqa_output_path)
        print(f"✓ NISQA 评估完成，处理了 {len(df_nisqa)} 个音频文件")
        
        # 整合结果
        print(f"\n整合评估结果...")
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
                    # TTS 模型只输出 Naturalness
                    result['naturalness'] = float(row.get('mos_pred', row.get('noi_pred', 0)))
                else:
                    # NISQA v2.0 输出多个维度
                    mos = float(row.get('mos_pred', 0))
                    result['mos'] = mos
                    result['noisiness'] = float(row.get('noi_pred', 0))
                    result['coloration'] = float(row.get('col_pred', 0))
                    result['discontinuity'] = float(row.get('dis_pred', 0))
                    result['loudness'] = float(row.get('loud_pred', 0))
                    
                    # 添加质量等级分类
                    if mos >= 4.0:
                        result['quality_grade'] = 'Excellent'
                    elif mos >= 3.5:
                        result['quality_grade'] = 'Good'
                    elif mos >= 3.0:
                        result['quality_grade'] = 'Fair'
                    else:
                        result['quality_grade'] = 'Poor'
                
                results.append(result)
        
        # 保存结果
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('video_id').reset_index(drop=True)
            df_results.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"\n{'=' * 80}")
            print("评估完成")
            print(f"{'=' * 80}")
            
            print(f"\n成功评估的视频: {len(results)}/{len(video_files)}")
            print(f"结果已保存到: {output_path}")
            
            # 统计分析
            print(f"\n{'=' * 80}")
            print("统计分析（评分范围：1-5，越高越好）")
            print(f"{'=' * 80}")
            
            if use_tts_model:
                print(f"\nNaturalness (自然度) 统计:")
                nat_scores = df_results['naturalness'].values
                print(f"  平均值: {nat_scores.mean():.4f}")
                print(f"  标准差: {nat_scores.std():.4f}")
                print(f"  最小值: {nat_scores.min():.4f}")
                print(f"  最大值: {nat_scores.max():.4f}")
                print(f"  中位数: {np.median(nat_scores):.4f}")
                
                # 显示自然度最高和最低的视频
                print(f"\n自然度最高的前10个视频:")
                top_10 = df_results.nlargest(10, 'naturalness')
                for idx, row in top_10.iterrows():
                    print(f"  {row['video_id']}: {row['naturalness']:.3f}")
                
                print(f"\n自然度最低的前10个视频:")
                bottom_10 = df_results.nsmallest(10, 'naturalness')
                for idx, row in bottom_10.iterrows():
                    print(f"  {row['video_id']}: {row['naturalness']:.3f}")
            
            else:
                print(f"\nMOS (整体质量) 统计:")
                mos_scores = df_results['mos'].values
                print(f"  平均值: {mos_scores.mean():.3f}")
                print(f"  标准差: {mos_scores.std():.3f}")
                print(f"  最小值: {mos_scores.min():.3f}")
                print(f"  最大值: {mos_scores.max():.3f}")
                print(f"  中位数: {np.median(mos_scores):.3f}")
                
                print(f"\n质量维度平均分:")
                print(f"  噪音度 (NOI): {df_results['noisiness'].mean():.3f} (越高噪声越少)")
                print(f"  色调失真 (COL): {df_results['coloration'].mean():.3f} (越高音色越自然)")
                print(f"  不连续性 (DIS): {df_results['discontinuity'].mean():.3f} (越高越连续)")
                print(f"  响度 (LOUD): {df_results['loudness'].mean():.3f} (越高响度越合适)")
                
                print(f"\n质量等级分布:")
                grade_counts = df_results['quality_grade'].value_counts()
                for grade in ['Excellent', 'Good', 'Fair', 'Poor']:
                    count = grade_counts.get(grade, 0)
                    pct = count / len(df_results) * 100
                    print(f"  {grade}: {count} ({pct:.1f}%)")
                
                print(f"\n质量最好的前10个视频（按MOS）:")
                best = df_results.nlargest(10, 'mos')
                for idx, row in best.iterrows():
                    print(f"  {row['video_id']}: MOS={row['mos']:.3f}, {row['quality_grade']}")
                
                print(f"\n质量最差的前10个视频（按MOS）:")
                worst = df_results.nsmallest(10, 'mos')
                for idx, row in worst.iterrows():
                    print(f"  {row['video_id']}: MOS={row['mos']:.3f}, {row['quality_grade']}")
                
                # 维度问题分析
                print(f"\n问题分析（基于质量维度）:")
                
                # 噪音问题
                noisy = df_results[df_results['noisiness'] < 3.0]
                if len(noisy) > 0:
                    print(f"  噪音问题 (NOI<3.0): {len(noisy)} 个视频")
                    print(f"    最严重: {noisy.nsmallest(3, 'noisiness')['video_id'].tolist()}")
                
                # 不连续性问题
                discontinuous = df_results[df_results['discontinuity'] < 3.0]
                if len(discontinuous) > 0:
                    print(f"  不连续性问题 (DIS<3.0): {len(discontinuous)} 个视频")
                    print(f"    最严重: {discontinuous.nsmallest(3, 'discontinuity')['video_id'].tolist()}")
                
                # 色调失真
                colored = df_results[df_results['coloration'] < 3.0]
                if len(colored) > 0:
                    print(f"  色调失真问题 (COL<3.0): {len(colored)} 个视频")
                    print(f"    最严重: {colored.nsmallest(3, 'coloration')['video_id'].tolist()}")
                
                # 响度问题
                loud_issue = df_results[df_results['loudness'] < 3.0]
                if len(loud_issue) > 0:
                    print(f"  响度问题 (LOUD<3.0): {len(loud_issue)} 个视频")
                    print(f"    最严重: {loud_issue.nsmallest(3, 'loudness')['video_id'].tolist()}")
        
        else:
            print(f"\n{'=' * 80}")
            print("评估完成")
            print(f"{'=' * 80}")
            print("\n没有成功评估的视频")
            df_results = pd.DataFrame()

    finally:
        print(f"\n清理临时音频文件...")
        import shutil
        try:
            shutil.rmtree(audio_dir)
        except Exception as e:
            print(f"警告: 清理临时目录失败: {e}")

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
        print(f"未发现可评估视频目录: {args.video_root}")
        return

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*80}")
        print(f"# 数据集: {label}  ({video_dir})")
        print(f"{'#'*80}")
        if not os.path.exists(video_dir):
            print("跳过：目录不存在")
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
    print(f"汇总结果已写入: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
