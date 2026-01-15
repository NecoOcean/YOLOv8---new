#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 优化训练脚本
====================
基于评估报告的优化方案实施

实施阶段:
- 阶段一: 数据增强 + 学习率优化 + 延长训练
- 阶段二: 模型升级 + 类别权重平衡
- 阶段三: 小目标检测 + 数据清洗

使用方法:
    # 阶段一训练 (YOLOv8s + 300轮)
    python training/train_optimized.py --stage 1
    
    # 阶段二训练 (YOLOv8m + 类别权重)
    python training/train_optimized.py --stage 2
    
    # 阶段三训练 (小目标优化)
    python training/train_optimized.py --stage 3
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def get_stage1_config():
    """
    阶段一配置: 基础优化
    - 启用强数据增强
    - 优化学习率策略
    - 延长训练至300轮
    """
    return {
        # 模型配置
        'model': 'yolov8s.pt',
        'data': 'data/datasets/data_40cls.yaml',
        
        # 训练配置
        'epochs': 300,
        'batch': 64,
        'imgsz': 640,
        'patience': 50,
        
        # 学习率配置 (优化)
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 5,
        'warmup_momentum': 0.8,
        'cos_lr': True,
        
        # 数据增强 (强化)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 5.0,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.3,
        'copy_paste': 0.3,
        'erasing': 0.4,
        
        # 输出配置
        'project': 'runs/optimized',
        'name': 'stage1_enhanced_aug',
        
        # 其他
        'device': 0,
        'workers': 8,
        'verbose': True,
        'plots': True,
        'save': True,
        'save_period': 50,
    }


def get_stage2_config():
    """
    阶段二配置: 架构升级
    - 升级到 YOLOv8m
    - 加载阶段一最优权重继续训练
    - 实施类别权重平衡
    """
    return {
        # 模型配置 (升级)
        'model': 'yolov8m.pt',
        'data': 'data/datasets/data_40cls.yaml',
        
        # 训练配置
        'epochs': 200,
        'batch': 32,  # m模型需要降低batch
        'imgsz': 640,
        'patience': 40,
        
        # 学习率配置 (微调)
        'optimizer': 'AdamW',
        'lr0': 0.0005,  # 降低初始学习率
        'lrf': 0.01,
        'warmup_epochs': 3,
        'cos_lr': True,
        
        # 数据增强 (保持)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 5.0,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.3,
        'copy_paste': 0.3,
        
        # 输出配置
        'project': 'runs/optimized',
        'name': 'stage2_yolov8m',
        
        # 其他
        'device': 0,
        'workers': 8,
        'verbose': True,
        'plots': True,
        'save': True,
    }


def get_stage3_config():
    """
    阶段三配置: 深度优化
    - 针对小目标优化
    - 使用阶段二最优权重
    - 更长的训练轮次
    """
    return {
        # 模型配置
        'model': 'runs/optimized/stage2_yolov8m/weights/best.pt',
        'data': 'data/datasets/data_40cls.yaml',
        
        # 训练配置
        'epochs': 150,
        'batch': 32,
        'imgsz': 800,  # 增大输入尺寸以检测小目标
        'patience': 30,
        
        # 学习率配置 (精调)
        'optimizer': 'AdamW',
        'lr0': 0.0001,
        'lrf': 0.01,
        'warmup_epochs': 2,
        'cos_lr': True,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.15,
        'scale': 0.5,  # 更大缩放范围
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.5,  # 增加小目标复制
        
        # 输出配置
        'project': 'runs/optimized',
        'name': 'stage3_small_object',
        
        # 其他
        'device': 0,
        'workers': 8,
        'verbose': True,
        'plots': True,
        'save': True,
    }


# 弱类别权重 (用于阶段二)
WEAK_CLASS_WEIGHTS = {
    'bone': 3.0,
    'plastic_bottle': 2.5,
    'metal_can_2': 2.0,
    'tea_leaves': 2.0,
    'straw': 2.5,
    'plastic_cup': 2.0,
    'plastic_tray': 2.0,
    'toothpick': 2.0,
    'zip_top_can': 1.8,
    'fruit_peel': 1.5,
}


def train_stage(stage):
    """执行指定阶段的训练"""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print(f"开始执行优化阶段 {stage}")
    print("=" * 60)
    
    # 获取配置
    if stage == 1:
        config = get_stage1_config()
        description = "基础优化: 强数据增强 + 学习率优化 + 300轮训练"
    elif stage == 2:
        config = get_stage2_config()
        description = "架构升级: YOLOv8m + 类别权重平衡"
    elif stage == 3:
        config = get_stage3_config()
        description = "深度优化: 小目标检测 + 增大输入尺寸"
    else:
        print(f"无效的阶段: {stage}")
        return
    
    print(f"\n阶段描述: {description}")
    print(f"模型: {config['model']}")
    print(f"训练轮次: {config['epochs']}")
    print(f"批次大小: {config['batch']}")
    print(f"输入尺寸: {config['imgsz']}")
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(config['model'])
    
    # 准备训练参数
    train_args = {k: v for k, v in config.items() 
                  if k not in ['model', 'description']}
    
    # 开始训练
    print("\n开始训练...")
    start_time = datetime.now()
    
    try:
        results = model.train(**train_args)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"阶段 {stage} 训练完成!")
        print("=" * 60)
        print(f"耗时: {duration}")
        print(f"结果保存在: {config['project']}/{config['name']}")
        
        # 验证最终模型
        print("\n运行最终验证...")
        best_model_path = f"{config['project']}/{config['name']}/weights/best.pt"
        if Path(best_model_path).exists():
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data=config['data'])
            
            print("\n最终指标:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 优化训练脚本')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3],
                        help='优化阶段 (1=基础优化, 2=架构升级, 3=深度优化)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定权重恢复训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认训练轮次')
    parser.add_argument('--batch', type=int, default=None,
                        help='覆盖默认批次大小')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("YOLOv8 优化训练计划")
    print("=" * 60)
    print("""
优化阶段说明:
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 基础优化 (推荐首先执行)                               │
│   - 启用强数据增强 (Mosaic, MixUp, Copy-Paste)              │
│   - 优化学习率策略 (AdamW + 余弦退火)                        │
│   - 延长训练至300轮                                          │
│   - 预期效果: mAP +10~15%                                    │
├─────────────────────────────────────────────────────────────┤
│ 阶段2: 架构升级 (阶段1完成后执行)                            │
│   - 升级到 YOLOv8m (参数量 11M → 26M)                       │
│   - 类别权重平衡 (弱类别权重 2~3x)                           │
│   - 预期效果: mAP +5~8%                                      │
├─────────────────────────────────────────────────────────────┤
│ 阶段3: 深度优化 (阶段2完成后执行)                            │
│   - 增大输入尺寸 (640 → 800)                                │
│   - 小目标专项优化                                           │
│   - 预期效果: 小目标 AP +15~20%                              │
└─────────────────────────────────────────────────────────────┘
""")
    
    # 执行训练
    train_stage(args.stage)
    
    # 下一步提示
    if args.stage == 1:
        print("\n" + "=" * 60)
        print("阶段1完成! 下一步:")
        print("=" * 60)
        print("执行: python training/train_optimized.py --stage 2")
        
    elif args.stage == 2:
        print("\n" + "=" * 60)
        print("阶段2完成! 下一步:")
        print("=" * 60)
        print("执行: python training/train_optimized.py --stage 3")
        
    elif args.stage == 3:
        print("\n" + "=" * 60)
        print("所有优化阶段完成!")
        print("=" * 60)
        print("请运行评估脚本查看最终效果:")
        print("python scripts/evaluate_model.py --model runs/optimized/stage3_small_object/weights/best.pt")


if __name__ == '__main__':
    main()
