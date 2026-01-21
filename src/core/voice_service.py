# src/core/voice_service.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 语音播报服务模块
"""
import pyttsx3
from typing import List, Set
from collections import defaultdict
import threading
import time
import queue


class VoiceService:
    """语音播报服务类 - 使用单一工作线程避免线程冲突"""
    
    def __init__(self):
        self.engine = None
        self.enabled = True
        self.announcement_interval = 2.0  # 每2秒最多播报一次
        self.last_announcement_time = 0
        self.last_announced_categories = set()
        
        # 创建消息队列和工作线程
        self.message_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.engine_ready = threading.Event()  # 用于同步 engine 初始化
        
        # 启动工作线程（engine 将在工作线程中初始化）
        self._start_worker()
        
        # 等待 engine 初始化完成
        if not self.engine_ready.wait(timeout=5.0):
            print("[WARNING] 语音引擎初始化超时")
            self.enabled = False
    
    def _init_engine(self):
        """初始化语音引擎（必须在工作线程中调用）"""
        try:
            # 关键修复：在 Windows 上，pyttsx3 必须在创建它的线程中使用
            print(f"[INFO] 在线程 {threading.current_thread().name} 中初始化语音引擎")
            self.engine = pyttsx3.init()
            # 设置语音属性
            self.engine.setProperty('rate', 150)  # 语速
            self.engine.setProperty('volume', 0.9)  # 音量
            print("[INFO] 语音播报引擎初始化成功")
            self.enabled = True
            self.engine_ready.set()  # 通知主线程 engine 已就绪
        except Exception as e:
            print(f"[WARNING] 语音播报引擎初始化失败: {e}")
            self.enabled = False
            self.engine_ready.set()  # 即使失败也要通知
    
    def _reinit_engine(self):
        """重新初始化引擎（解决 Windows COM 状态问题）"""
        try:
            if self.engine:
                try:
                    self.engine.stop()
                except:
                    pass
                del self.engine
                self.engine = None
            
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            print("[DEBUG] 引擎已重新初始化")
        except Exception as e:
            print(f"[ERROR] 重新初始化引擎失败: {e}")
            self.engine = None
    
    def _start_worker(self):
        """启动工作线程"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=False, name="VoiceWorker")
        self.worker_thread.start()
        print("[INFO] 语音播报工作线程已启动")
    
    def _worker_loop(self):
        """工作线程主循环 - 处理所有语音播报请求"""
        # 关键修复：在工作线程中初始化 engine
        self._init_engine()
        
        if not self.engine:
            print("[ERROR] 语音引擎初始化失败，工作线程退出")
            return
        
        print("[INFO] 语音工作线程开始处理消息")
        
        while self.running:
            try:
                # 从队列获取消息，超时0.5秒则继续循环
                text = self.message_queue.get(timeout=0.5)
                
                if text is None:  # None 是停止信号
                    break
                
                # 执行语音播报
                try:
                    print(f"[VOICE] {text}")
                    if self.engine and self.enabled:
                        self.engine.say(text)
                        self.engine.runAndWait()
                        print(f"[DEBUG] 播报完成: {text[:30]}...")
                        # 关键修复：每次播报后重新初始化引擎
                        # 彻底解决 Windows SAPI5 COM 对象状态问题
                        self._reinit_engine()
                except Exception as e:
                    print(f"[WARNING] 语音播报失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 尝试重新初始化引擎
                    self._reinit_engine()
                
                # 标记任务完成
                self.message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] 工作线程异常: {e}")
                import traceback
                traceback.print_exc()
        
        print("[INFO] 语音播报工作线程已停止")
    
    def set_enabled(self, enabled: bool):
        """启用/禁用语音播报"""
        self.enabled = enabled
    
    def is_enabled(self) -> bool:
        """检查是否启用语音播报"""
        return self.enabled and self.engine is not None
    
    def announce_detection(self, detection_result, announce_all: bool = False):
        """
        播报检测结果
        
        Args:
            detection_result: 检测结果对象
            announce_all: 是否播报所有检测到的物品（默认只播报分类）
        """
        if not self.is_enabled() or not detection_result.has_detections:
            return
        
        # 防止频繁播报（时间间隔控制）
        current_time = time.time()
        if current_time - self.last_announcement_time < self.announcement_interval:
            print(f"[DEBUG] 播报被间隔限制，距上次 {current_time - self.last_announcement_time:.1f}秒")
            return
        
        # 获取分类指导信息
        guides = detection_result.get_classification_guide()
        
        # 按类别分组
        category_items = defaultdict(list)
        for guide in guides:
            category = guide.get('category', '未知')
            item_name = guide.get('name', '')
            category_items[category].append(item_name)
        
        # 更新上次播报的类别和时间
        current_categories = set(category_items.keys())
        self.last_announced_categories = current_categories
        self.last_announcement_time = current_time
        
        # 生成播报文本
        announcement_text = self._generate_announcement(category_items, announce_all)
        
        # 将消息放入队列，由工作线程处理
        if announcement_text:
            try:
                self.message_queue.put(announcement_text, block=False)
                print(f"[DEBUG] 语音消息已加入队列: {announcement_text[:30]}...")
            except queue.Full:
                print("[WARNING] 语音消息队列已满，跳过此次播报")
    
    def _generate_announcement(self, category_items: dict, announce_all: bool) -> str:
        """
        生成播报文本
        
        Args:
            category_items: 按类别分组的物品字典
            announce_all: 是否播报所有物品名称
        """
        announcements = []
        
        # 定义垃圾桶类型映射
        bin_mapping = {
            '厨余垃圾': '绿色厨余垃圾桶',
            '可回收物': '蓝色可回收物垃圾桶',
            '有害垃圾': '红色有害垃圾桶',
            '其他垃圾': '灰色其他垃圾桶',
        }
        
        for category, items in category_items.items():
            bin_name = bin_mapping.get(category, '对应垃圾桶')
            
            if announce_all and items:
                # 播报具体物品
                items_text = '、'.join(set(items[:3]))  # 最多播报3个不重复的物品
                if len(items) > 3:
                    items_text += '等'
                announcements.append(f"检测到{items_text}，请投入{bin_name}")
            else:
                # 只播报类别
                announcements.append(f"检测到{category}，请投入{bin_name}")
        
        # 如果检测到多个类别
        if len(announcements) > 1:
            return "检测到多种垃圾。" + "；".join(announcements)
        elif announcements:
            return announcements[0]
        else:
            return ""
    
    def _speak(self, text: str):
        """
        直接播报文本（内部使用，由工作线程调用）
        注意：该方法现在由 _worker_loop 调用，不再需要
        """
        # 该方法保留以保持向后兼容，但不再使用
        pass
    
    def announce_text(self, text: str):
        """
        直接播报指定文本
        
        Args:
            text: 要播报的文本
        """
        if not self.is_enabled() or not text:
            return
        
        try:
            self.message_queue.put(text, block=False)
            print(f"[DEBUG] 自定义文本已加入队列: {text[:30]}...")
        except queue.Full:
            print("[WARNING] 语音消息队列已满，跳过此次播报")
    
    def set_rate(self, rate: int):
        """
        设置语速
        
        Args:
            rate: 语速值，通常在100-200之间
        """
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        设置音量
        
        Args:
            volume: 音量值，0.0-1.0之间
        """
        if self.engine:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def set_announcement_interval(self, interval: float):
        """
        设置播报间隔
        
        Args:
            interval: 间隔秒数
        """
        self.announcement_interval = max(1.0, interval)
    
    def reset_announcement_cache(self):
        """重置播报缓存（用于切换图片/视频时）"""
        self.last_announced_categories = set()
        # 设置为当前时间减去间隔，确保下次检测立即播报
        self.last_announcement_time = time.time() - self.announcement_interval - 0.1
    
    def stop(self):
        """停止当前播报和工作线程"""
        print("[INFO] 正在停止语音服务...")
        
        # 发送停止信号
        self.running = False
        
        # 清空队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break
        
        # 放入 None 作为停止信号
        try:
            self.message_queue.put(None, block=False)
        except queue.Full:
            pass
        
        # 等待工作线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        # 停止引擎
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        print("[INFO] 语音服务已停止")
    
    def __del__(self):
        """析构函数"""
        self.stop()
