"""
265类垃圾分类模型训练脚本
数据集: tany0699/garbage265 (转换后)
- 147,286 张训练/验证图片
- 265 个中文类别
"""

from ultralytics import YOLO
from pathlib import Path
import yaml


def train_265cls():
    """训练265类垃圾分类模型"""
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 数据集配置文件路径
    data_yaml = project_root / "data/datasets/tany0699_yolo/data.yaml"
    
    if not data_yaml.exists():
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        print("请先运行: python scripts/convert_tany0699.py")
        return
    
    # 验证数据集
    print(f"数据集配置: {data_yaml}")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"类别数量: {config.get('nc', 'N/A')}")
    
    # 加载预训练模型
    model_path = project_root / "data/models/pretrained/yolov8n.pt"
    if not model_path.exists():
        print(f"预训练模型不存在，将自动下载: yolov8n.pt")
        model = YOLO("yolov8n.pt")
    else:
        print(f"加载预训练模型: {model_path}")
        model = YOLO(str(model_path))
    
    # 训练参数
    train_args = {
        "data": str(data_yaml),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,  # 根据GPU显存调整
        "patience": 20,  # 早停
        "device": 0,  # GPU
        "workers": 8,
        "project": str(project_root / "runs/detect"),
        "name": "garbage_265cls",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5,
        "cos_lr": True,
        "close_mosaic": 10,  # 最后10个epoch关闭mosaic
        "amp": True,  # 混合精度训练
        "cache": False,  # 数据集较大，不缓存到内存
        "verbose": True,
    }
    
    print("\n" + "="*50)
    print("开始训练 265 类垃圾分类模型")
    print("="*50)
    print(f"训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # 开始训练
    results = model.train(**train_args)
    
    # 训练完成
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)
    print(f"最佳模型: {project_root}/runs/detect/garbage_265cls/weights/best.pt")
    print(f"最终模型: {project_root}/runs/detect/garbage_265cls/weights/last.pt")
    
    # 复制最佳模型到 models 目录
    best_model_src = project_root / "runs/detect/garbage_265cls/weights/best.pt"
    best_model_dst = project_root / "data/models/trained/best_265cls.pt"
    if best_model_src.exists():
        import shutil
        best_model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_src, best_model_dst)
        print(f"已复制模型到: {best_model_dst}")
    
    return results


if __name__ == "__main__":
    train_265cls()
