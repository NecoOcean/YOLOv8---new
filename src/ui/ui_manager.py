# src/ui/ui_manager.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - UI管理模块
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from src.config import settings


class UIManager:
    """UI管理器类"""
    
    def __init__(self, ui):
        self.ui = ui
    
    def display_image(self, label: QLabel, image: np.ndarray):
        """在QLabel上显示图像"""
        if image is None:
            return
        
        # 转换颜色空间
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
        # 获取标签大小
        label_width = label.width()
        label_height = label.height()
        
        # 等比例缩放
        h, w = image.shape[:2]
        scale = min(label_width / w, label_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为QPixmap
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
    
    def update_result_table(self, table: QTableWidget, detection_result):
        """更新检测结果表格"""
        config = settings.get_current_config()
        table.setRowCount(0)
        
        if not detection_result.has_detections:
            return
        
        for i, cls_id in enumerate(detection_result.classes):
            row = table.rowCount()
            table.insertRow(row)
            
            # 类别名称
            ch_name = config['CH_names'][cls_id] if cls_id < len(config['CH_names']) else f'类别{cls_id}'
            table.setItem(row, 0, QTableWidgetItem(ch_name))
            
            # 置信度
            table.setItem(row, 1, QTableWidgetItem(detection_result.confidence_strings[i]))
            
            # 位置
            loc = detection_result.locations[i]
            loc_str = f"({loc[0]},{loc[1]})-({loc[2]},{loc[3]})"
            table.setItem(row, 2, QTableWidgetItem(loc_str))
            
            # 分类
            guide = config['classification_guide'].get(cls_id, {})
            table.setItem(row, 3, QTableWidgetItem(guide.get('category', '未知')))
    
    def update_detection_info(self, label: QLabel, count: int, elapsed_time: float):
        """更新检测信息标签"""
        text = f"检测到 {count} 个目标 | 耗时: {elapsed_time*1000:.1f}ms"
        label.setText(text)
    
    def show_classification_guide(self, label: QLabel, guides: list):
        """显示分类指导"""
        if not guides:
            label.setText("无检测结果")
            return
        
        # 按类别分组
        category_tips = {}
        for guide in guides:
            cat = guide.get('category', '未知')
            if cat not in category_tips:
                category_tips[cat] = {
                    'color': guide.get('color', 'gray'),
                    'tip': guide.get('tip', ''),
                    'items': []
                }
            category_tips[cat]['items'].append(guide.get('name', ''))
        
        # 生成HTML
        html_parts = []
        color_map = {
            'green': '#4CAF50',
            'blue': '#2196F3',
            'red': '#F44336',
            'gray': '#9E9E9E'
        }
        
        for cat, info in category_tips.items():
            color = color_map.get(info['color'], '#9E9E9E')
            items_str = '、'.join(info['items'])
            html_parts.append(
                f'<p><span style="color:{color};font-weight:bold;">【{cat}】</span>'
                f'{items_str}<br/>'
                f'<span style="color:#666;">{info["tip"]}</span></p>'
            )
        
        label.setText(''.join(html_parts))
    
    def clear_display(self, label: QLabel):
        """清空显示"""
        label.clear()
        label.setText("")
    
    def set_status(self, label: QLabel, message: str):
        """设置状态信息"""
        label.setText(message)
    
    def update_combobox(self, combobox: QComboBox, items: list, current_index: int = 0):
        """更新下拉框"""
        combobox.clear()
        combobox.addItems(items)
        if 0 <= current_index < len(items):
            combobox.setCurrentIndex(current_index)
    
    def update_statistics_display(self, stats: dict):
        """更新统计显示"""
        if hasattr(self.ui, 'todayStatsLabel'):
            today = stats.get('date', '')
            count = stats.get('detection_count', 0)
            items = stats.get('total_items', 0)
            self.ui.todayStatsLabel.setText(f"今日: {count}次检测, {items}个目标")
        
        breakdown = stats.get('category_breakdown', {})
        
        if hasattr(self.ui, 'kitchenLabel'):
            self.ui.kitchenLabel.setText(f"厨余垃圾: {breakdown.get('厨余垃圾', 0)}")
        if hasattr(self.ui, 'recyclableLabel'):
            self.ui.recyclableLabel.setText(f"可回收物: {breakdown.get('可回收物', 0)}")
        if hasattr(self.ui, 'hazardousLabel'):
            self.ui.hazardousLabel.setText(f"有害垃圾: {breakdown.get('有害垃圾', 0)}")
        if hasattr(self.ui, 'otherLabel'):
            self.ui.otherLabel.setText(f"其他垃圾: {breakdown.get('其他垃圾', 0)}")
