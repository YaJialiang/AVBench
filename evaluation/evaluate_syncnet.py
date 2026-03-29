#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate lip-sync quality of AVBench videos with SyncNet (LatentSync)."""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import cv2

from common_paths import default_results_root, default_video_root, discover_video_dirs


def find_video_files(video_dir: str) -> List[str]:
    """查找目录下的所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    return sorted([str(f) for f in video_files])


def extract_video_id(video_path: str) -> str:
    """从视频文件名中提取 ID"""
    filename = Path(video_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        video_id = f"{parts[0]}_{parts[1]}"
        return video_id
    return filename


def compute_sync_score(confidence: float, offset_frames: int) -> float:
    """
    综合唇语同步评分（0-100）

    设计原则：
    - 置信度高 + 偏移量小 → 分数最高
    - 置信度高 + 偏移量大 → 分数最低（乘法耦合使高置信度更敏感于偏移）
    - 置信度低 + 偏移量小 → 中低分
    - 置信度低 + 偏移量大 → 低分

    公式：
      conf_score   = sigmoid((conf - 5.0) * 0.5)   # 置信度归一化，中心5分
      offset_decay = exp(-|offset_frames| / 3.0)    # 偏移指数衰减，3帧≈半衰
      sync_score   = conf_score * offset_decay * 100
    """
    import math
    conf_score   = 1.0 / (1.0 + math.exp(-(confidence - 5.0) * 0.5))
    offset_decay = math.exp(-abs(offset_frames) / 3.0)
    return round(conf_score * offset_decay * 100.0, 2)


def evaluate_lip_sync(video_path: str, 
                     syncnet_eval: SyncNetEval,
                     temp_dir: str,
                     batch_size: int = 20,
                     vshift: int = 15) -> Dict:
    """
    评估单个视频的唇语同步度
    
    Args:
        video_path: 视频路径
        syncnet_eval: SyncNet评估器
        temp_dir: 临时目录
        batch_size: 批次大小
        vshift: 最大偏移帧数
        
    Returns:
        评估结果字典
    """
    try:
        # 读取原始 fps 仅用于元数据记录，不影响评估逻辑
        # syncnet_eval 内部抽帧时强制使用 25fps，保证与训练分布一致
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if src_fps <= 0:
            src_fps = 25.0

        # 运行 SyncNet 评估
        # evaluate() 内部：-vf fps=25 抽帧 + 固定 25fps MFCC 对齐
        # 返回值：(offset_frames, min_dist, confidence, src_fps_read_by_syncnet)
        offset, min_dist, conf, _ = syncnet_eval.evaluate(
            video_path,
            temp_dir=temp_dir,
            batch_size=batch_size,
            vshift=vshift,
        )

        # 转换为 Python 标量
        offset   = int(offset)
        conf     = float(conf)
        min_dist = float(min_dist)

        # offset_sec 固定基于 25fps（帧索引含义统一，跨模型可比）
        EVAL_FPS  = 25.0
        offset_sec = offset / EVAL_FPS
        
        # 评估同步质量
        if conf >= 7.0:
            sync_quality = 'Excellent'
        elif conf >= 5.0:
            sync_quality = 'Good'
        elif conf >= 3.0:
            sync_quality = 'Fair'
        else:
            sync_quality = 'Poor'

        # 综合评分
        sync_score = compute_sync_score(conf, offset)

        return {
            'offset_frames': offset,
            'offset_sec': offset_sec,
            'src_fps': round(src_fps, 3),
            'confidence': conf,
            'min_dist': min_dist,
            'sync_quality': sync_quality,
            'sync_score': sync_score,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'offset_frames': None,
            'offset_sec': None,
            'confidence': None,
            'min_dist': None,
            'sync_quality': None,
            'success': False,
            'error': str(e)
        }


def evaluate_videos(video_dir: str, 
                    output_path: str,
                    latentsync_root: str,
                    syncnet_ckpt: str,
                    device: str = 'cuda',
                    batch_size: int = 20,
                    vshift: int = 15):
    """
    评估所有视频的唇语同步度
    
    Args:
        video_dir: 视频目录
        output_path: 输出CSV文件路径
        syncnet_ckpt: SyncNet检查点路径
        device: 运行设备
        batch_size: 批次大小
        vshift: 最大偏移帧数
    """
    print("=" * 80)
    print("SyncNet 唇语同步度评估")
    print("=" * 80)
    
    # 查找所有视频文件
    video_files = find_video_files(video_dir)
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        print(f"错误: 在 {video_dir} 中未找到视频文件")
        return
    
    # Initialize SyncNet from a configurable LatentSync repository path.
    print(f"\n正在初始化 SyncNet 模型...")
    sys.path.insert(0, latentsync_root)
    from eval.syncnet.syncnet_eval import SyncNetEval
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    syncnet_eval = SyncNetEval(device=device)
    
    # 加载预训练权重
    if os.path.exists(syncnet_ckpt):
        checkpoint = torch.load(syncnet_ckpt, map_location=device)
        
        # 处理不同格式的checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        syncnet_eval.__S__.load_state_dict(state_dict, strict=False)
        print(f"SyncNet 模型加载成功: {syncnet_ckpt}")
    else:
        print(f"警告: 未找到SyncNet权重文件: {syncnet_ckpt}")
        print("将使用未初始化的模型（结果可能不准确）")
    
    syncnet_eval.__S__.eval()
    
    # 创建临时目录
    temp_base_dir = "/tmp/syncnet_eval"
    os.makedirs(temp_base_dir, exist_ok=True)
    
    # 评估结果
    results = []
    failed_videos = []
    
    print(f"\n开始评估唇语同步度...")
    print(f"参数: batch_size={batch_size}, vshift={vshift} 帧（偏移时间由各视频实际 FPS 决定）")
    
    for video_path in tqdm(video_files, desc="评估进度"):
        video_id = extract_video_id(video_path)
        video_name = Path(video_path).name
        
        # 为每个视频创建独立的临时目录
        temp_dir = os.path.join(temp_base_dir, video_id)
        
        try:
            # 评估唇语同步
            result = evaluate_lip_sync(
                video_path, 
                syncnet_eval, 
                temp_dir,
                batch_size=batch_size,
                vshift=vshift
            )
            
            if result['success']:
                results.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'src_fps': result['src_fps'],   # 原始帧率（模型生成质量维度）
                    'offset_frames': result['offset_frames'],   # 基于 25fps
                    'offset_sec': result['offset_sec'],         # 基于 25fps，跨模型可比
                    'confidence': result['confidence'],
                    'min_dist': result['min_dist'],
                    'sync_quality': result['sync_quality'],
                    'sync_score': result['sync_score'],
                })
            else:
                failed_videos.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'error': result['error']
                })
                print(f"\n错误: 评估 {video_name} 失败: {result['error']}")
            
        except Exception as e:
            print(f"\n错误: 评估 {video_name} 失败: {str(e)}")
            failed_videos.append({
                'video_id': video_id,
                'video_name': video_name,
                'error': str(e)
            })
    
    # 清理临时目录
    import shutil
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir, ignore_errors=True)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('video_id')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n评估完成，结果已保存到: {output_path}")
        return df
    else:
        print("\n没有成功评估的视频")
        return pd.DataFrame()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lip-sync quality with SyncNet.")
    parser.add_argument("--video-root", default=str(default_video_root()), help="Root directory of video datasets.")
    parser.add_argument("--results-dir", default=str(default_results_root()), help="Directory to save CSV outputs.")
    parser.add_argument("--latentsync-root", default=os.environ.get("LATENTSYNC_ROOT", "/public/yangjl/LatentSync"), help="Path to LatentSync repository.")
    parser.add_argument("--syncnet-ckpt", default=None, help="Path to syncnet_v2.model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    latentsync_root = args.latentsync_root
    syncnet_ckpt = args.syncnet_ckpt or os.path.join(latentsync_root, "checkpoints", "auxiliary", "syncnet_v2.model")
    device = args.device
    batch_size = args.batch_size
    vshift = args.vshift
    results_dir  = args.results_dir
    video_dirs = discover_video_dirs(Path(args.video_root))
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
        output_path = os.path.join(results_dir, f"syncnet_{label}.csv")
        df = evaluate_videos(
            video_dir    = video_dir,
            latentsync_root = latentsync_root,
            output_path  = output_path,
            syncnet_ckpt = syncnet_ckpt,
            device       = device,
            batch_size   = batch_size,
            vshift       = vshift,
        )
        if df is not None and not df.empty:
            summary_rows.append({
                "dataset":         label,
                "n_videos":        len(df),
                "mean_confidence": round(float(df["confidence"].mean()), 4),
                "mean_offset_sec": round(float(df["offset_sec"].mean()), 4),
                "mean_abs_offset": round(float(df["offset_sec"].abs().mean()), 4),
                "mean_sync_score": round(float(df["sync_score"].mean()), 4),
            })

    summary_path = os.path.join(results_dir, "syncnet_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"汇总结果已写入: {summary_path}")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
