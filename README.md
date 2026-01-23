# 基于YOLOv8的厨房垃圾目标检测系统

## 项目简介

本项目是一个基于YOLOv8深度学习目标检测算法的智能垃圾分类识别系统，应用于智能家居场景（厨房环境），实现垃圾的自动识别、分类指导以及语音提示。

## 核心功能

- **多模态检测**：支持单张图片、文件夹批量、视频文件及摄像头实时检测。
- **4类智能分类**：精准识别厨余垃圾、可回收物、有害垃圾及其他垃圾。
- **语音播报**：集成智能语音提示，自动播报垃圾分类投放指导。
- **统计管理**：自动记录检测历史，支持按日统计及数据导出（CSV）。
- **模块化架构**：采用 src/ 目录结构的现代化 Python 项目架构，易于扩展。

## 技术栈

- **深度学习**：Ultralytics YOLOv8
- **GUI框架**：PyQt5
- **图像处理**：OpenCV
- **语音合成**：pyttsx3
- **编程语言**：Python 3.9+

## 快速开始

### 1. 环境准备

```bash
# 创建并激活虚拟环境
conda create -n garbage_detect python=3.9
conda activate garbage_detect

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行程序

```bash
# 运行主应用程序 (重构版入口)
python app.py
```

## 项目结构

```
YOLOv8/
├── app.py                  # 应用程序主入口
├── src/                    # 源代码目录
│   ├── config/             # 配置管理 (settings.py)
│   ├── core/               # 核心逻辑 (检测服务、统计、语音)
│   ├── ui/                 # UI管理及样式
│   └── utils/              # 工具函数 (文件处理)
├── data/                   # 数据资源目录
│   ├── models/trained/     # 训练好的模型 (.pt)
│   └── datasets/           # 数据集配置 (.yaml)
├── output/                 # 输出目录 (检测结果、统计报表)
└── training/               # 训练脚本目录
```

## 详细文档

- [项目架构说明](docs/项目架构说明.md)
- [操作手册](docs/操作手册.md)
- [4类模型部署指南](docs/4类模型部署指南.md)
- [语音播报功能说明](docs/语音播报功能说明.md)

## 许可证

MIT License
