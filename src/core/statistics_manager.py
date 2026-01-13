# src/core/statistics_manager.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 统计管理模块
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from src.config import settings


class StatisticsManager:
    """统计管理器类"""
    
    def __init__(self, save_path: str = None):
        self.save_path = Path(save_path) if save_path else settings.STATISTICS_DIR
        self.statistics_file = self.save_path / 'statistics.json'
        self.records = []
        self._load_records()
    
    def _load_records(self):
        """加载历史记录"""
        if self.statistics_file.exists():
            try:
                with open(self.statistics_file, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = []
        else:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.records = []
    
    def _save_records(self):
        """保存记录到文件"""
        self.save_path.mkdir(parents=True, exist_ok=True)
        with open(self.statistics_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
    
    def add_record(self, detection_result, source_info: str = ''):
        """添加检测记录"""
        config = settings.get_current_config()
        now = datetime.now()
        
        items = []
        for i, cls_id in enumerate(detection_result.classes):
            ch_name = config['CH_names'][cls_id] if cls_id < len(config['CH_names']) else f'类别{cls_id}'
            guide = config['classification_guide'].get(cls_id, {})
            items.append({
                'class_id': cls_id,
                'name': ch_name,
                'category': guide.get('category', '未知'),
                'confidence': round(detection_result.confidences[i], 4)
            })
        
        record = {
            'id': len(self.records) + 1,
            'timestamp': now.isoformat(),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'source': source_info,
            'total_count': detection_result.count,
            'items': items,
            'elapsed_time': round(detection_result.elapsed_time * 1000, 2)  # 毫秒
        }
        
        self.records.append(record)
        self._save_records()
        return record
    
    def get_today_statistics(self) -> Dict:
        """获取今日统计"""
        today = datetime.now().strftime('%Y-%m-%d')
        today_records = [r for r in self.records if r.get('date') == today]
        
        total_items = sum(r.get('total_count', 0) for r in today_records)
        
        category_breakdown = defaultdict(int)
        for record in today_records:
            for item in record.get('items', []):
                category_breakdown[item.get('category', '未知')] += 1
        
        return {
            'date': today,
            'detection_count': len(today_records),
            'total_items': total_items,
            'category_breakdown': dict(category_breakdown)
        }
    
    def get_category_statistics(self) -> Dict[str, int]:
        """获取分类统计"""
        stats = defaultdict(int)
        for record in self.records:
            for item in record.get('items', []):
                stats[item.get('category', '未知')] += 1
        return dict(stats)
    
    def get_class_statistics(self) -> Dict[str, int]:
        """获取类别统计"""
        stats = defaultdict(int)
        for record in self.records:
            for item in record.get('items', []):
                stats[item.get('name', '未知')] += 1
        return dict(stats)
    
    def get_daily_statistics(self, days: int = 7) -> List[Dict]:
        """获取每日统计"""
        from datetime import timedelta
        
        result = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            day_records = [r for r in self.records if r.get('date') == date]
            result.append({
                'date': date,
                'detection_count': len(day_records),
                'total_items': sum(r.get('total_count', 0) for r in day_records)
            })
        return result
    
    def export_to_csv(self, output_path: str = None) -> str:
        """导出为CSV文件"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = settings.EXPORTS_DIR / f'statistics_{timestamp}.csv'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['记录ID', '日期', '时间', '检测数量', '类别详情', '垃圾分类', '检测耗时(ms)'])
            
            for record in self.records:
                names = ';'.join([item['name'] for item in record.get('items', [])])
                categories = ';'.join([item['category'] for item in record.get('items', [])])
                writer.writerow([
                    record.get('id'),
                    record.get('date'),
                    record.get('time'),
                    record.get('total_count'),
                    names,
                    categories,
                    record.get('elapsed_time')
                ])
        
        return str(output_path)
    
    def clear_records(self):
        """清空所有记录"""
        self.records = []
        self._save_records()
    
    @property
    def total_records(self) -> int:
        """总记录数"""
        return len(self.records)
