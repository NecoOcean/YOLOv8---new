# training/train.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 统一训练入口
支持5类和40类模型训练
"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model(mode: str = 'cls5', epochs: int = 100, batch: int = 16, 
                device: str = '0', resume: bool = False):
    """
    训练模型
    
    Args:
        mode: 训练模式 'cls5' 或 'cls40'
        epochs: 训练轮次
        batch: 批次大小
        device: 设备 '0' GPU或 'cpu'
        resume: 是否继续训练
    """
    
    # 配置
    if mode == 'cls40':
        data_yaml = str(PROJECT_ROOT / 'data' / 'datasets' / 'data_40cls.yaml')
        project_name = 'garbage_40cls'
        base_model = 'yolov8s.pt'
    else:
        data_yaml = str(PROJECT_ROOT / 'data' / 'datasets' / 'kitchen_garbage' / 'data.yaml')
        project_name = 'kitchen_garbage_5cls'
        base_model = 'yolov8n.pt'
    
    print(f"\n{'='*50}")
    print(f"训练模式: {mode.upper()}")
    print(f"数据集配置: {data_yaml}")
    print(f"基础模型: {base_model}")
    print(f"训练轮次: {epochs}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    print(f"{'='*50}\n")
    
    # 加载模型
    model = YOLO(base_model)
    
    # 训练配置 (优化后的数据增强策略)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        resume=resume,
        project=str(PROJECT_ROOT / 'training' / 'runs' / 'detect'),
        name=project_name,
        
        # ========== 优化的学习率策略 ==========
        optimizer='AdamW',      # 比 Adam 更稳定
        lr0=0.001,              # 初始学习率
        lrf=0.01,               # 最终学习率比例
        cos_lr=True,            # 余弦退火
        warmup_epochs=5,        # 预热轮次
        warmup_momentum=0.8,    # 预热动量
        
        # ========== 训练控制 ==========
        patience=50,            # 早停耐心值 (增加)
        save=True,
        save_period=50,         # 每50轮保存
        
        # ========== 强化数据增强策略 ==========
        # 颜色增强
        hsv_h=0.015,            # 色调变化
        hsv_s=0.7,              # 饱和度变化
        hsv_v=0.4,              # 亮度变化
        
        # 几何变换 (增强)
        degrees=15.0,           # 旋转角度 (10→15)
        translate=0.2,          # 平移比例 (0.1→0.2)
        scale=0.9,              # 缩放范围 (0.5→0.9)
        shear=5.0,              # 剪切角度 (新增)
        perspective=0.001,      # 透视变换 (新增)
        
        # 翻转
        flipud=0.5,             # 垂直翻转 (新增)
        fliplr=0.5,             # 水平翻转
        
        # 高级增强 (新增)
        mosaic=1.0,             # Mosaic增强
        mixup=0.3,              # MixUp增强 (新增)
        copy_paste=0.3,         # Copy-Paste (新增，针对小目标)
        erasing=0.4,            # 随机擦除 (新增)
        
        # 其他
        workers=8,              # 数据加载线程
        verbose=True,
        plots=True,
    )
    
    # 验证
    print("\n" + "="*50)
    print("模型验证结果:")
    print("="*50)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 输出路径
    output_dir = PROJECT_ROOT / 'training' / 'runs' / 'detect' / project_name
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {output_dir / 'weights' / 'best.pt'}")
    print(f"\n请执行以下命令复制模型到data/models/trained目录:")
    
    if mode == 'cls40':
        print(f"copy \"{output_dir / 'weights' / 'best.pt'}\" \"{PROJECT_ROOT / 'data' / 'models' / 'trained' / 'best_40cls.pt'}\"")
    else:
        print(f"copy \"{output_dir / 'weights' / 'best.pt'}\" \"{PROJECT_ROOT / 'data' / 'models' / 'trained' / 'best_5cls.pt'}\"")


def main():
    parser = argparse.ArgumentParser(description='垃圾检测模型训练 (优化版)')
    parser.add_argument('--mode', type=str, default='cls5', choices=['cls5', 'cls40'],
                        help='训练模式: cls5(5类) 或 cls40(40类)')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮次 (默认300)')
    parser.add_argument('--batch', type=int, default=64, help='批次大小 (默认64)')
    parser.add_argument('--device', type=str, default='0', help='设备: 0/1/cpu')
    parser.add_argument('--resume', action='store_true', help='继续训练')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("数据增强策略已优化:")
    print("=" * 60)
    print("✅ 学习率: AdamW + 余弦退火 + 预热")
    print("✅ 几何增强: 旋转15° + 平移20% + 剪切 + 透视")
    print("✅ 翻转: 水平50% + 垂直50%")
    print("✅ 高级增强: Mosaic + MixUp(30%) + Copy-Paste(30%)")
    print("✅ 其他: 随机擦除(40%)")
    print("=" * 60)
    
    train_model(
        mode=args.mode,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
