"""Evaluate AVBench videos with DOVER++ for aesthetic/technical/overall quality."""

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
import torch
import yaml

from common_paths import default_results_root, default_video_root, discover_video_dirs

warnings.filterwarnings('ignore')


def fuse_results(results: list):
    """
    融合技术质量和美学质量得分
    Args:
        results: [technical_score, aesthetic_score]
    Returns:
        dict with aesthetic, technical, and overall scores (0-100)
    """
    # results[0]: technical, results[1]: aesthetic
    t, a = (results[0] - 0.1107) / 0.07355, (results[1] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)) * 100,
        "technical": 1 / (1 + np.exp(-t)) * 100,
        "overall": 1 / (1 + np.exp(-x)) * 100,
    }


def classify_quality(score: float) -> str:
    """根据分数分类视频质量"""
    if score >= 75:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 45:
        return 'Fair'
    else:
        return 'Poor'


def find_video_files(video_dir: str) -> List[str]:
    """查找目录下的所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    return sorted([str(f) for f in video_files])


def extract_video_id(video_path: str) -> str:
    """从视频文件名中提取ID"""
    filename = Path(video_path).stem
    # 提取 voice_XXXX 部分
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return filename


def evaluate_videos_dover(
    video_dir: str,
    dover_config: str,
    model_path: str,
    dover_root: str,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    使用 DOVER++ 评估视频目录中的所有视频
    
    Args:
        video_dir: 视频目录路径
        dover_config: DOVER配置文件路径
        model_path: DOVER++模型权重路径
        device: 运行设备 (cuda/cpu)
    
    Returns:
        包含评估结果的 DataFrame
    """
    print("=" * 60)
    print("DOVER++ 视频质量评估")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"配置文件: {dover_config}")
    print(f"模型权重: {model_path}")
    print(f"运行设备: {device}")
    print()
    
    # Lazy import so users can customize DOVER location.
    sys.path.insert(0, dover_root)
    from dover.datasets import ViewDecompositionDataset
    from dover.models import DOVER

    # 加载配置
    with open(dover_config, "r") as f:
        opt = yaml.safe_load(f)
    
    # 加载DOVER++模型
    print("加载 DOVER++ 模型...")
    evaluator = DOVER(**opt["model"]["args"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # DOVER++权重文件包含state_dict键
    if 'state_dict' in checkpoint:
        evaluator.load_state_dict(checkpoint['state_dict'])
    else:
        evaluator.load_state_dict(checkpoint)
    evaluator.eval()
    print("✓ 模型加载完成\n")
    
    # 查找视频文件
    video_files = find_video_files(video_dir)
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    if not video_files:
        print("错误：未找到视频文件！")
        return pd.DataFrame()
    
    # 准备数据加载器
    dopt = opt["data"]["val-l1080p"]["args"].copy()
    dopt["anno_file"] = None
    dopt["data_prefix"] = video_dir
    
    dataset = ViewDecompositionDataset(dopt)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=opt.get("num_workers", 4), 
        pin_memory=True,
    )
    
    results_list = []
    sample_types = ["technical", "aesthetic"]
    
    print("开始评估视频...")
    for i, data in enumerate(tqdm(dataloader, desc="评估进度")):
        if len(data.keys()) == 1:
            # 加载失败的数据
            continue
        
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )
        
        with torch.no_grad():
            raw_results = evaluator(video, reduce_scores=False)
            raw_results = [np.mean(l.cpu().numpy()) for l in raw_results]
        
        # 融合结果
        scores = fuse_results(raw_results)
        
        # 提取视频信息
        video_name = data["name"][0]
        video_id = extract_video_id(video_name)
        
        results_list.append({
            'video_id': video_id,
            'video_name': os.path.basename(video_name),
            'aesthetic_score': scores['aesthetic'],
            'technical_score': scores['technical'],
            'overall_score': scores['overall'],
            'aesthetic_grade': classify_quality(scores['aesthetic']),
            'technical_grade': classify_quality(scores['technical']),
            'overall_grade': classify_quality(scores['overall']),
        })
    
    # 创建DataFrame
    df = pd.DataFrame(results_list)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"总视频数: {len(df)}")
    print()
    
    print("技术质量 (VT值) 统计:")
    print(f"  平均分: {df['technical_score'].mean():.2f}")
    print(f"  标准差: {df['technical_score'].std():.2f}")
    print(f"  最高分: {df['technical_score'].max():.2f} ({df.loc[df['technical_score'].idxmax(), 'video_id']})")
    print(f"  最低分: {df['technical_score'].min():.2f} ({df.loc[df['technical_score'].idxmin(), 'video_id']})")
    print()
    
    print("技术质量分布:")
    tech_dist = df['technical_grade'].value_counts()
    for grade in ['Excellent', 'Good', 'Fair', 'Poor']:
        count = tech_dist.get(grade, 0)
        pct = count / len(df) * 100
        print(f"  {grade:10s}: {count:2d} ({pct:5.1f}%)")
    print()
    
    print("美学质量统计:")
    print(f"  平均分: {df['aesthetic_score'].mean():.2f}")
    print(f"  标准差: {df['aesthetic_score'].std():.2f}")
    print(f"  最高分: {df['aesthetic_score'].max():.2f} ({df.loc[df['aesthetic_score'].idxmax(), 'video_id']})")
    print(f"  最低分: {df['aesthetic_score'].min():.2f} ({df.loc[df['aesthetic_score'].idxmin(), 'video_id']})")
    print()
    
    print("整体质量统计:")
    print(f"  平均分: {df['overall_score'].mean():.2f}")
    print(f"  标准差: {df['overall_score'].std():.2f}")
    print(f"  最高分: {df['overall_score'].max():.2f} ({df.loc[df['overall_score'].idxmax(), 'video_id']})")
    print(f"  最低分: {df['overall_score'].min():.2f} ({df.loc[df['overall_score'].idxmin(), 'video_id']})")
    print()
    
    # 分析问题视频
    poor_tech = df[df['technical_score'] < 45]
    if len(poor_tech) > 0:
        print(f"技术质量较差的视频 (VT<45): {len(poor_tech)} 个")
        print("  这些视频可能存在以下问题：")
        print("  - 清晰度不足")
        print("  - 锐利度较差")
        print("  - 渲染失真或伪影")
        print()
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run DOVER++ quality evaluation for AVBench videos.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--dover-root", default=os.environ.get("DOVER_ROOT", "/public/yangjl/DOVER"), help="Path to DOVER repository.")
    parser.add_argument("--dover-config", default=None, help="Path to dover.yml. Defaults to <dover-root>/dover.yml")
    parser.add_argument("--model-path", default=None, help="Path to DOVER++ model checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    dover_root = args.dover_root
    dover_config = args.dover_config or os.path.join(dover_root, "dover.yml")
    model_path = args.model_path or os.path.join(dover_root, "pretrained_weights", "DOVER_plus_plus.pth")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir  = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
    os.makedirs(results_dir, exist_ok=True)

    if not video_dirs:
        print(f"未发现可评估视频目录: {args.video_root}")
        return

    if not os.path.exists(model_path):
        print(f"错误：找不到DOVER++模型权重文件: {model_path}")
        print("请运行：wget https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth")
        return

    summary_rows = []
    for label, video_dir in video_dirs:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {label}  ({video_dir})")
        print(f"{'#'*60}")
        if not os.path.exists(video_dir):
            print("跳过：目录不存在")
            continue
        df = evaluate_videos_dover(video_dir, dover_config, model_path, dover_root, device)
        if not df.empty:
            output_csv = os.path.join(results_dir, f"dover_{label}.csv")
            df.to_csv(output_csv, index=False)
            print(f"逐视频结果已保存: {output_csv}")
            summary_rows.append({
                "dataset":        label,
                "n_videos":       len(df),
                "mean_aesthetic": round(df["aesthetic_score"].mean(), 4),
                "mean_technical": round(df["technical_score"].mean(), 4),
                "mean_overall":   round(df["overall_score"].mean(), 4),
            })

    summary_path = os.path.join(results_dir, "dover_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"汇总结果已写入: {summary_path}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
