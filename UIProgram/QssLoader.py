# QssLoader.py
# -*- coding: utf-8 -*-
"""
QSS样式加载器
"""
import os


class QSSLoader:
    """QSS样式表加载器"""
    
    @staticmethod
    def read_qss_file(qss_file_path: str) -> str:
        """读取QSS文件"""
        if not os.path.exists(qss_file_path):
            print(f"[WARNING] QSS文件不存在: {qss_file_path}")
            return ""
        
        try:
            with open(qss_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[ERROR] 读取QSS文件失败: {e}")
            return ""
    
    @staticmethod
    def load_qss(widget, qss_file_path: str) -> bool:
        """为控件加载QSS样式"""
        qss_content = QSSLoader.read_qss_file(qss_file_path)
        if qss_content:
            widget.setStyleSheet(qss_content)
            return True
        return False
