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
    
    # 训练配置
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        cos_lr=True,
        optimizer='Adam',
        device=device,
        patience=20,
        save=True,
        project=str(PROJECT_ROOT / 'training' / 'runs' / 'detect'),
        name=project_name,
        resume=resume,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
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
    parser = argparse.ArgumentParser(description='垃圾检测模型训练')
    parser.add_argument('--mode', type=str, default='cls5', choices=['cls5', 'cls40'],
                        help='训练模式: cls5(5类) 或 cls40(40类)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='设备: 0/1/cpu')
    parser.add_argument('--resume', action='store_true', help='继续训练')
    
    args = parser.parse_args()
    
    train_model(
        mode=args.mode,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
