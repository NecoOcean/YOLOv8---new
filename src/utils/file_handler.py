# src/utils/file_handler.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 文件处理模块
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

from src.config import settings


class FileHandler:
    """文件处理工具类"""
    
    # 支持的图片格式
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
    
    # 支持的视频格式
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """判断是否为图片文件"""
        ext = Path(file_path).suffix.lower()
        return ext in FileHandler.IMAGE_EXTENSIONS
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """判断是否为视频文件"""
        ext = Path(file_path).suffix.lower()
        return ext in FileHandler.VIDEO_EXTENSIONS
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """获取文件类型"""
        if FileHandler.is_image_file(file_path):
            return 'image'
        elif FileHandler.is_video_file(file_path):
            return 'video'
        return 'unknown'
    
    @staticmethod
    def get_images_from_directory(directory: str) -> List[str]:
        """从目录获取所有图片文件"""
        images = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return images
        
        for file in dir_path.iterdir():
            if file.is_file() and FileHandler.is_image_file(str(file)):
                images.append(str(file))
        
        return sorted(images)
    
    @staticmethod
    def generate_save_path(original_path: str, save_dir: str = None, 
                          prefix: str = 'detected_') -> str:
        """生成保存路径"""
        if save_dir is None:
            save_dir = str(settings.DETECTIONS_DIR / 'images')
        
        # 确保目录存在
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        original_name = Path(original_path).stem
        ext = Path(original_path).suffix or '.jpg'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        new_name = f"{prefix}{original_name}_{timestamp}{ext}"
        return str(Path(save_dir) / new_name)
    
    @staticmethod
    def generate_video_save_path(original_path: str = None, 
                                 save_dir: str = None) -> str:
        """生成视频保存路径"""
        if save_dir is None:
            save_dir = str(settings.DETECTIONS_DIR / 'videos')
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if original_path:
            original_name = Path(original_path).stem
            new_name = f"detected_{original_name}_{timestamp}.mp4"
        else:
            new_name = f"camera_{timestamp}.mp4"
        
        return str(Path(save_dir) / new_name)
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """确保目录存在"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[dict]:
        """获取文件信息"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_str': FileHandler.format_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_image': FileHandler.is_image_file(file_path),
            'is_video': FileHandler.is_video_file(file_path)
        }
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def get_unique_path(path: str) -> str:
        """获取唯一路径（如果存在则添加序号）"""
        original_path = Path(path)
        if not original_path.exists():
            return path
        
        stem = original_path.stem
        suffix = original_path.suffix
        parent = original_path.parent
        
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1
