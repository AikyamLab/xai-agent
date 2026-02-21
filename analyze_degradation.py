import os
import json
from collections import defaultdict
import numpy as np

def analyze_degradation(root_dir):
    # 数据结构：stats[modality][dataset] = {"orig": [], "impr": []}
    stats = defaultdict(lambda: defaultdict(lambda: {"orig": [], "impr": []}))

    for root, dirs, files in os.walk(root_dir):
        if "training_datapoint.json" in files:
            file_path = os.path.join(root, "training_datapoint.json")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 检查是否有改进尝试
                metrics = data.get("improvement_metrics")
                if not metrics:
                    continue
                
                delta = metrics.get("faithfulness_delta", 0)
                
                # 只统计变差 (delta < 0) 或持平 (delta == 0) 的情况
                if delta > 0:
                    modality = data.get("modality", "unknown")
                    # 从路径中提取数据集名称更准确，或者从 question 对象中拿
                    dataset = data.get("question", {}).get("dataset", "unknown")
                    if isinstance(dataset, list): # 处理 Adult Census 这种 list 格式
                        dataset = dataset[0]
                    
                    # 获取原始分数和改进后的分数
                    orig_score = data.get("original", {}).get("explanation_faithfulness", {}).get("score")
                    impr_score = data.get("improved", {}).get("explanation_faithfulness", {}).get("score")
                    
                    # 只有当两个分数都存在且为数值时才记录
                    if orig_score is not None and impr_score is not None:
                        stats[modality][dataset]["orig"].append(float(orig_score))
                        stats[modality][dataset]["impr"].append(float(impr_score))
                    
            except Exception as e:
                continue

    return stats

def print_stats(stats):
    print(f"{'Modality':<10} | {'Dataset':<20} | {'Count':<6} | {'Original (Min-Max / Mean)':<30} | {'Improved (Min-Max / Mean)':<30}")
    print("-" * 110)
    
    for mod in sorted(stats.keys()):
        for ds in sorted(stats[mod].keys()):
            orig = stats[mod][ds]["orig"]
            impr = stats[mod][ds]["impr"]
            count = len(orig)
            
            if count == 0: continue
            
            orig_min, orig_max, orig_mean = min(orig), max(orig), np.mean(orig)
            impr_min, impr_max, impr_mean = min(impr), max(impr), np.mean(impr)
            
            orig_str = f"{orig_min:.4f}-{orig_max:.4f} / {orig_mean:.4f}"
            impr_str = f"{impr_min:.4f}-{impr_max:.4f} / {impr_mean:.4f}"
            
            print(f"{mod:<10} | {ds:<20} | {count:<6} | {orig_str:<30} | {impr_str:<30}")

if __name__ == "__main__":
    root_path = "outputs/training_data"
    degradation_stats = analyze_degradation(root_path)
    print_stats(degradation_stats)
