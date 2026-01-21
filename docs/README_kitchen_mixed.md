# Kitchen Mixed 数据集训练指南

## 数据集概述

`kitchen_mixed` 是一个综合性的厨房垃圾分类数据集，包含 **4个类别**：

| 类别ID | 名称 | 中文 | 描述 |
|--------|------|------|------|
| 0 | kitchen_waste | 厨余垃圾 | 蔬菜废料、果皮、骨头等 |
| 1 | recyclable | 可回收物 | 塑料瓶、易拉罐、纸盒、玻璃等 |
| 2 | hazardous | 有害垃圾 | 药品、针管、医疗废物等 |
| 3 | other | 其他垃圾 | 烟蒂等不可分类垃圾 |

### 数据来源
- **TU Wien Domestic Organic Waste** - 家庭有机厨余垃圾
- **Medical Waste 1 & 2** - Roboflow 医疗废物数据集
- **TACO 10 Class** - 垃圾分类检测数据集

## 快速开始

### 方式1：使用统一训练入口

```bash
# 使用 cls4 模式 (4类)
python training/train.py --mode cls4 --epochs 100 --batch 32

# 使用 mixed 模式 (更强的模型)
python training/train.py --mode mixed --epochs 150 --batch 16
```

### 方式2：使用专用训练脚本

```bash
# 基础训练
python training/train_kitchen_mixed.py

# 自定义参数
python training/train_kitchen_mixed.py --model yolov8s --epochs 200 --batch 32
```

## 训练参数详解

### 模型选择

| 模型 | 参数量 | 推荐批次 | 适用场景 |
|------|--------|----------|----------|
| yolov8n | 3.2M | 64 | 边缘设备、实时检测 |
| yolov8s | 11.2M | 32 | **推荐**，平衡速度与精度 |
| yolov8m | 25.9M | 16 | 更高精度需求 |
| yolov8l | 43.7M | 8 | 高精度场景 |
| yolov8x | 68.2M | 4 | 最高精度 |

### 数据增强策略

```bash
# 轻量增强 - 适合小数据集
python training/train_kitchen_mixed.py --augment light

# 标准增强 - 推荐
python training/train_kitchen_mixed.py --augment standard

# 强化增强 - 防止过拟合
python training/train_kitchen_mixed.py --augment aggressive
```

### 学习率配置

```bash
# 默认学习率
python training/train_kitchen_mixed.py --lr0 0.001 --lrf 0.01

# 更小的学习率 (精细调优)
python training/train_kitchen_mixed.py --lr0 0.0005 --lrf 0.001
```

## autoDL 云服务器训练

### 推荐配置

```bash
# 单GPU (RTX 3090 / A100)
python training/train_kitchen_mixed.py \
    --model yolov8s \
    --epochs 200 \
    --batch 32 \
    --device 0 \
    --augment standard

# 多GPU
python training/train_kitchen_mixed.py \
    --model yolov8m \
    --epochs 200 \
    --batch 64 \
    --device 0,1
```

### 注意事项

1. 训练结果自动保存到 `/root/autodl-tmp/training_runs`
2. 脚本会自动检测 autoDL 环境并启用优化
3. 启用了混合精度训练 (AMP) 加速
4. 数据会缓存到 RAM 提高 IO 效率

## 继续训练

```bash
# 从中断处继续
python training/train_kitchen_mixed.py --resume

# 从特定权重继续
python training/train_kitchen_mixed.py \
    --resume \
    --weights /path/to/last.pt
```

## 训练输出

训练完成后，模型自动保存到以下位置：

```
training/runs/kitchen_mixed_yolov8s_20240122_120000/
├── weights/
│   ├── best.pt        # 最佳模型
│   └── last.pt        # 最新模型
├── results.csv        # 训练指标
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── ...
```

最佳模型会自动复制到：
```
data/models/trained/best_mixed.pt
```

## 常见问题

### Q: 显存不足怎么办？
A: 减小 batch size 或使用更小的模型：
```bash
python training/train_kitchen_mixed.py --batch 8 --model yolov8n
```

### Q: 训练太慢？
A: 使用更小的模型或启用缓存：
```bash
python training/train_kitchen_mixed.py --model yolov8n --batch 64
```

### Q: 如何评估模型？
A: 训练完成后会自动验证，也可以手动运行：
```bash
python scripts/evaluate_model.py --weights data/models/trained/best_mixed.pt
```
