# 基于YOLOv8的垃圾目标检测系统

> ⚠️ **重要提示**：项目已于2026年1月13日完成架构重构，详细结构请参阅 [项目架构说明.md](项目架构说明.md)

## 项目简介

本项目是一个基于YOLOv8深度学习目标检测算法的智能垃圾分类识别系统，应用于智能家居场景（厨房环境），实现垃圾的自动识别与分类指导。

## 快速开始

```bash
# 运行应用（推荐新入口）
python app.py

# 或使用旧版入口
python main.py
```

## 配置模式

项目支持两种分类精度：

| 模式 | 类别数 | 适用场景 |
|------|--------|----------|
| **简化版** | 5类 | 快速演示、轻量部署 |
| **精细版** | 40类 | 完整功能、精确识别 |

**切换命令**：

```bash
python scripts/switch_config.py 5    # 切换到5类配置
python scripts/switch_config.py 40   # 切换到40类配置
```

> 详细架构说明请参阅 [项目架构说明.md](项目架构说明.md)

## 功能特点

- **图片检测**：单张图片垃圾识别
- **批量检测**：文件夹批量处理
- **视频检测**：视频文件逐帧检测
- **摄像头检测**：实时画面检测
- **分类指导**：显示垃圾分类建议（厨余/可回收/有害/其他）
- **结果保存**：保存检测结果图片/视频

## 技术栈

- **深度学习框架**：Ultralytics YOLOv8
- **GUI框架**：PyQt5
- **图像处理**：OpenCV
- **编程语言**：Python 3.8+

## 项目结构

```
YOLOv8/
├── main.py                 # 主程序入口
├── detection_service.py    # 检测服务模块
├── ui_manager.py           # UI管理模块
├── file_handler.py         # 文件处理模块
├── statistics_manager.py   # 统计管理模块
├── Config.py               # 当前使用的配置文件
├── Config_kitchen.py       # 5类简化配置
├── Config_40cls.py         # 40类精细化配置
├── train.py                # 5类模型训练脚本
├── train_40cls.py          # 40类模型训练脚本
├── switch_config.py        # 配置切换工具
├── requirements.txt        # 依赖列表
├── UIProgram/              # UI界面模块
│   ├── UiMain.py           # 主窗口UI定义
│   ├── QssLoader.py        # 样式加载器
│   └── style.css           # 样式表
├── models/                 # 模型文件目录
├── datasets/               # 数据集目录
│   ├── data_40cls.yaml     # 40类数据集配置
│   └── kitchen_garbage/    # 5类数据集配置
├── save_data/              # 检测结果保存目录
└── TestFiles/              # 测试文件目录
```

## 安装步骤

### 1. 创建虚拟环境

```bash
conda create -n garbage_detect python=3.9
conda activate garbage_detect
```

### 2. 安装PyTorch（GPU版本）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 5类模型训练（简化版）
python train.py
# 训练完成后复制: runs/detect/kitchen_garbage_5cls/weights/best.pt -> models/best.pt

# 40类模型训练（精细版）
python train_40cls.py
# 训练完成后复制: runs/detect/garbage_40cls/weights/best.pt -> models/best_40cls.pt
```

### 运行应用程序

```bash
python main.py
```

### 快捷键

- `Ctrl+O`：打开图片
- `Ctrl+Q`：退出程序
- `←/→`：切换图片（批量模式）
- `Esc`：停止视频/摄像头

## 数据集

项目包含两种数据集配置：

### 5类简化版（kitchen_garbage）
- 类别：果皮、茶叶渣、易拉罐、过期药品、其他垃圾
- 配置文件：`datasets/kitchen_garbage/data.yaml`

### 40类精细版
- 类别：涵盖塑料、厨余、纸类、金属、玻璃等40种细分类别
- 配置文件：`datasets/data_40cls.yaml`
- 数据来源：原始40类标注数据（16,840张训练图片，1,776张验证图片）

```
datasets/
├── images/
│   ├── train/           # 训练集图片 (16,840张)
│   └── val/             # 验证集图片 (1,776张)
├── labels/
│   ├── train/           # 训练集标注 (19,028个)
│   └── val/             # 验证集标注
├── data_40cls.yaml      # 40类数据集配置
└── kitchen_garbage/     # 5类简化数据集
    └── data.yaml
```

## 许可证

MIT License
