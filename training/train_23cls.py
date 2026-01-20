"""
23类厨房垃圾分类模型训练脚本
数据集: TACO + Food Waste Detection + tany0699 (混合)
- 156,407 张训练/验证图片
- 23 个目标类别（厨余/可回收/有害/其他）
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
from datetime import datetime


def train_23cls():
    """训练23类厨房垃圾分类模型"""
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 数据集配置文件路径
    data_yaml = project_root / "data/datasets/kitchen_garbage_merged_v2/data.yaml"
    
    if not data_yaml.exists():
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        print("请先运行: python scripts/merge_datasets.py --tany0699 data/datasets/tany0699_yolo")
        return
    
    # 验证数据集
    print(f"数据集配置: {data_yaml}")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"类别数量: {config.get('nc', 'N/A')}")
    print(f"训练集: {config.get('train', 'N/A')}")
    print(f"验证集: {config.get('val', 'N/A')}")
    
    # 检查 GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 0
    else:
        print("警告: 未检测到 GPU，将使用 CPU 训练（速度较慢）")
        device = 'cpu'
    
    # 加载预训练模型
    model_path = project_root / "data/models/pretrained/yolov8n.pt"
    if not model_path.exists():
        print(f"预训练模型不存在，将自动下载: yolov8n.pt")
        model = YOLO("yolov8n.pt")
    else:
        print(f"加载预训练模型: {model_path}")
        model = YOLO(str(model_path))
    
    # 生成时间戳用于区分每次训练
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"kitchen_garbage_23cls_{timestamp}"
    
    # 训练参数（针对大数据集优化）
    train_args = {
        "data": str(data_yaml),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,  # 根据GPU显存调整，16GB显存可用batch=32
        "patience": 15,  # 早停（大数据集可适当减少）
        "device": device,
        "workers": 8,  # Windows 上可能需要设为 0
        "project": str(project_root / "training/runs"),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "cos_lr": True,
        "close_mosaic": 10,  # 最后10个epoch关闭mosaic
        "amp": True,  # 混合精度训练（节省显存）
        "cache": False,  # 数据集大，不缓存到内存
        "verbose": True,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 10,  # 每10个epoch保存一次
    }
    
    print("\n" + "="*60)
    print("开始训练 23 类厨房垃圾分类模型")
    print("="*60)
    print(f"训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 开始训练
    results = model.train(**train_args)
    
    # 训练完成
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    best_model = project_root / f"training/runs/{run_name}/weights/best.pt"
    last_model = project_root / f"training/runs/{run_name}/weights/last.pt"
    
    print(f"最佳模型: {best_model}")
    print(f"最终模型: {last_model}")
    
    # 复制最佳模型到 models 目录
    if best_model.exists():
        import shutil
        dst_model = project_root / "data/models/trained/best_23cls.pt"
        dst_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model, dst_model)
        print(f"已复制模型到: {dst_model}")
    
    # 验证最佳模型
    print("\n验证最佳模型...")
    val_results = model.val(data=str(data_yaml))
    
    print(f"\n验证结果:")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    
    return results


def train_with_resume(checkpoint_path: str = None):
    """从检查点恢复训练"""
    
    project_root = Path(__file__).parent.parent
    
    if checkpoint_path is None:
        checkpoint_path = project_root / "training/runs/kitchen_garbage_23cls/weights/last.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"错误: 检查点不存在: {checkpoint_path}")
        return
    
    print(f"从检查点恢复训练: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    
    results = model.train(resume=True)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练23类厨房垃圾分类模型')
    parser.add_argument('--resume', action='store_true',
                        help='从上次检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定检查点路径')
    
    args = parser.parse_args()
    
    if args.resume:
        train_with_resume(args.checkpoint)
    else:
        train_23cls()
