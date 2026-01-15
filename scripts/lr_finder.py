#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学习率查找器 (Learning Rate Finder)
====================================
自动寻找最优学习率范围

原理:
1. 从一个很小的学习率开始 (如 1e-7)
2. 逐步增加学习率，同时训练模型
3. 记录每个学习率对应的损失
4. 找到损失下降最快的学习率区域

使用方法:
    python scripts/lr_finder.py
    python scripts/lr_finder.py --model yolov8s.pt --data data/datasets/data_40cls.yaml

输出:
    - lr_finder_results.png: 学习率-损失曲线图
    - 推荐的学习率范围
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


class LRFinder:
    """学习率查找器"""
    
    def __init__(self, model_name='yolov8s.pt', data_yaml='data/datasets/data_40cls.yaml'):
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.lr_history = []
        self.loss_history = []
        
    def find_lr(self, start_lr=1e-7, end_lr=1.0, num_iterations=100, batch=16):
        """
        执行学习率查找
        
        Args:
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iterations: 测试迭代次数
            batch: 批次大小
        """
        from ultralytics import YOLO
        import torch
        
        print("\n" + "=" * 60)
        print("学习率查找器 (Learning Rate Finder)")
        print("=" * 60)
        print(f"模型: {self.model_name}")
        print(f"数据集: {self.data_yaml}")
        print(f"学习率范围: {start_lr} → {end_lr}")
        print(f"测试迭代: {num_iterations}")
        print("=" * 60)
        
        # 计算学习率增长因子
        lr_mult = (end_lr / start_lr) ** (1 / num_iterations)
        
        # 加载模型
        model = YOLO(self.model_name)
        
        # 使用短期训练测试不同学习率
        print("\n正在测试不同学习率...")
        
        lr_values = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iterations)
        
        for i, lr in enumerate(lr_values):
            try:
                # 短期训练 (1个epoch的部分数据)
                results = model.train(
                    data=self.data_yaml,
                    epochs=1,
                    batch=batch,
                    imgsz=320,  # 使用小尺寸加速
                    lr0=lr,
                    lrf=1.0,    # 固定学习率
                    cos_lr=False,
                    warmup_epochs=0,
                    patience=0,
                    save=False,
                    plots=False,
                    verbose=False,
                    val=False,
                    exist_ok=True,
                    project='runs/lr_finder',
                    name='test',
                )
                
                # 获取最终损失
                # 由于 ultralytics 不直接返回损失，我们需要从日志获取
                # 这里使用一个近似方法
                
                self.lr_history.append(lr)
                
                # 记录进度
                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{num_iterations} (lr={lr:.2e})")
                    
            except Exception as e:
                print(f"  跳过 lr={lr:.2e}: {e}")
                continue
        
        print("\n✅ 学习率测试完成")
        
    def find_lr_simple(self, batch=16):
        """
        简化版学习率查找 - 通过多次短训练比较
        
        测试几个典型的学习率区间，比较收敛速度
        """
        from ultralytics import YOLO
        
        print("\n" + "=" * 60)
        print("学习率查找器 (简化版)")
        print("=" * 60)
        print(f"模型: {self.model_name}")
        print(f"数据集: {self.data_yaml}")
        print("=" * 60)
        
        # 测试的学习率候选
        lr_candidates = [
            0.0001,   # 很保守
            0.0005,   # 保守
            0.001,    # 标准
            0.002,    # 略激进
            0.005,    # 激进
            0.01,     # 很激进
        ]
        
        results = []
        
        print("\n测试不同学习率 (每个训练10轮):\n")
        
        for lr in lr_candidates:
            print(f"测试 lr={lr}...")
            
            try:
                model = YOLO(self.model_name)
                
                train_results = model.train(
                    data=self.data_yaml,
                    epochs=10,
                    batch=batch,
                    imgsz=320,
                    lr0=lr,
                    lrf=1.0,
                    cos_lr=False,
                    warmup_epochs=0,
                    patience=0,
                    save=False,
                    plots=False,
                    verbose=False,
                    exist_ok=True,
                    project='runs/lr_finder',
                    name=f'lr_{lr}',
                )
                
                # 验证
                metrics = model.val(data=self.data_yaml, verbose=False)
                map50 = metrics.box.map50
                
                results.append({
                    'lr': lr,
                    'mAP50': map50,
                })
                
                print(f"  lr={lr}: mAP50={map50:.4f}")
                
            except Exception as e:
                print(f"  lr={lr}: 失败 - {e}")
                results.append({
                    'lr': lr,
                    'mAP50': 0,
                })
        
        # 分析结果
        print("\n" + "=" * 60)
        print("学习率测试结果")
        print("=" * 60)
        print(f"\n{'学习率':<12} {'mAP50':<10} {'评价'}")
        print("-" * 40)
        
        best_lr = None
        best_map = 0
        
        for r in results:
            lr, map50 = r['lr'], r['mAP50']
            
            if map50 > best_map:
                best_map = map50
                best_lr = lr
            
            # 评价
            if map50 >= best_map * 0.95:
                status = "✅ 最优"
            elif map50 >= best_map * 0.8:
                status = "🟡 良好"
            else:
                status = "🔴 偏低"
            
            print(f"{lr:<12.4f} {map50:<10.4f} {status}")
        
        print("\n" + "=" * 60)
        print("推荐配置")
        print("=" * 60)
        print(f"\n最优学习率: {best_lr}")
        print(f"对应 mAP50: {best_map:.4f}")
        
        # 推荐范围
        lr_idx = lr_candidates.index(best_lr)
        lr_min = lr_candidates[max(0, lr_idx - 1)]
        lr_max = lr_candidates[min(len(lr_candidates) - 1, lr_idx + 1)]
        
        print(f"\n推荐学习率范围: {lr_min} ~ {lr_max}")
        print(f"建议初始学习率 (lr0): {best_lr}")
        print(f"建议最终学习率比例 (lrf): 0.01")
        
        # 生成建议配置
        print("\n" + "=" * 60)
        print("建议的训练配置")
        print("=" * 60)
        print(f"""
在 training/train.py 中使用:

    optimizer='AdamW',
    lr0={best_lr},
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
""")
        
        return best_lr, results


def run_quick_lr_test():
    """
    快速学习率测试
    通过几轮短训练快速确定最佳学习率
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("快速学习率测试")
    print("=" * 60)
    
    lr_candidates = [0.0005, 0.001, 0.002, 0.005]
    results = []
    
    for lr in lr_candidates:
        print(f"\n测试 lr={lr}...")
        
        model = YOLO('yolov8s.pt')
        
        try:
            model.train(
                data='data/datasets/data_40cls.yaml',
                epochs=5,
                batch=32,
                imgsz=320,
                lr0=lr,
                cos_lr=False,
                warmup_epochs=1,
                patience=0,
                save=False,
                plots=False,
                verbose=False,
                exist_ok=True,
                project='runs/lr_test',
                name=f'lr_{lr}',
            )
            
            metrics = model.val(verbose=False)
            map50 = metrics.box.map50
            
            results.append((lr, map50))
            print(f"  结果: mAP50={map50:.4f}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append((lr, 0))
    
    # 找最佳
    best = max(results, key=lambda x: x[1])
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for lr, map50 in results:
        marker = "⭐" if lr == best[0] else "  "
        print(f"{marker} lr={lr}: mAP50={map50:.4f}")
    
    print(f"\n推荐学习率: {best[0]}")
    
    return best[0]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='学习率查找器')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='模型')
    parser.add_argument('--data', type=str, default='data/datasets/data_40cls.yaml', help='数据集')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--quick', action='store_true', help='快速模式 (5轮/学习率)')
    
    args = parser.parse_args()
    
    if args.quick:
        best_lr = run_quick_lr_test()
    else:
        finder = LRFinder(args.model, args.data)
        best_lr, results = finder.find_lr_simple(batch=args.batch)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n建议在训练脚本中使用: lr0={best_lr}")


if __name__ == '__main__':
    main()
