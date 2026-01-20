# app.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 应用程序入口（重构版）
"""
import sys
import os
import cv2
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon

# 导入重构后的模块
from src.config import settings
from src.core.detection_service import DetectionService, DetectionResult
from src.core.statistics_manager import StatisticsManager
from src.ui.ui_manager import UIManager
from src.utils.file_handler import FileHandler

# 导入UI（保持向后兼容）
from UIProgram.UiMain import Ui_MainWindow


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(object, object)  # (帧图像, 检测结果)
    finished_signal = pyqtSignal()
    
    def __init__(self, detection_service: DetectionService):
        super().__init__()
        self.detection_service = detection_service
        self.source = None
        self.running = False
        self.is_camera = False
    
    def set_source(self, source, is_camera: bool = False):
        self.source = source
        self.is_camera = is_camera
    
    def run(self):
        if self.source is None:
            return
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished_signal.emit()
            return
        
        self.running = True
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行检测
            result = self.detection_service.detect(frame)
            plotted_frame = result.get_plotted_image()
            
            self.frame_ready.emit(plotted_frame, result)
            
            # 控制帧率
            if not self.is_camera:
                self.msleep(33)  # 约30fps
        
        cap.release()
        self.finished_signal.emit()
    
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化核心组件
        self._init_components()
        
        # 连接信号槽
        self._connect_signals()
        
        # 初始状态
        self.current_image_path = None
        self.current_result = None
        self.image_list = []
        self.current_image_index = 0
        self.video_thread = None
        self.is_camera_running = False
        
        # 更新统计显示
        self._update_statistics_display()
        
        # 初始化配置菜单状态
        self._update_config_menu_state()
    
    def _init_components(self):
        """初始化核心组件"""
        # 检测服务
        try:
            # 尝试使用新配置路径
            model_path = str(settings.CURRENT_MODEL_PATH)
            if not Path(model_path).exists():
                # 回退到旧路径
                model_path = 'models/best.pt'
            self.detection_service = DetectionService(model_path)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"模型加载失败: {e}\n请确保模型文件存在")
            self.detection_service = None
        
        # UI管理器
        self.ui_manager = UIManager(self.ui)
        
        # 统计管理器
        self.statistics_manager = StatisticsManager()
    
    def _connect_signals(self):
        """连接信号槽"""
        # 按钮
        self.ui.PicBtn.clicked.connect(self.on_open_image)
        self.ui.FolderBtn.clicked.connect(self.on_open_folder)
        self.ui.VideoBtn.clicked.connect(self.on_open_video)
        self.ui.CapBtn.clicked.connect(self.on_toggle_camera)
        self.ui.SaveBtn.clicked.connect(self.on_save)
        self.ui.StopBtn.clicked.connect(self.on_stop)
        
        # 统计按钮
        if hasattr(self.ui, 'exportBtn'):
            self.ui.exportBtn.clicked.connect(self.on_export_statistics)
        if hasattr(self.ui, 'clearBtn'):
            self.ui.clearBtn.clicked.connect(self.on_clear_statistics)
        
        # 配置切换菜单
        if hasattr(self.ui, 'actionConfig5'):
            self.ui.actionConfig5.triggered.connect(lambda: self.on_switch_config('cls5'))
            self.ui.actionConfig23.triggered.connect(lambda: self.on_switch_config('cls23'))
            self.ui.actionConfig40.triggered.connect(lambda: self.on_switch_config('cls40'))
    
    def on_open_image(self):
        """打开图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif);;所有文件 (*.*)"
        )
        
        if file_path and self.detection_service:
            self.current_image_path = file_path
            self._detect_image(file_path)
    
    def on_open_folder(self):
        """打开文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        
        if folder_path:
            self.image_list = FileHandler.get_images_from_directory(folder_path)
            if self.image_list:
                self.current_image_index = 0
                self._detect_image(self.image_list[0])
                self.ui_manager.set_status(
                    self.ui.statusLabel, 
                    f"已加载 {len(self.image_list)} 张图片 (1/{len(self.image_list)})"
                )
    
    def on_open_video(self):
        """打开视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_path and self.detection_service:
            self._start_video(file_path)
    
    def on_toggle_camera(self):
        """切换摄像头"""
        if self.is_camera_running:
            self.on_stop()
        else:
            self._start_video(0, is_camera=True)
            self.is_camera_running = True
            self.ui.CapBtn.setText("关闭摄像头")
    
    def on_save(self):
        """保存结果"""
        if self.current_result and self.current_image_path:
            save_path = FileHandler.generate_save_path(self.current_image_path)
            plotted_image = self.current_result.get_plotted_image()
            cv2.imwrite(save_path, plotted_image)
            QMessageBox.information(self, "提示", f"已保存到: {save_path}")
    
    def on_stop(self):
        """停止"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.is_camera_running = False
        self.ui.CapBtn.setText("开启摄像头")
        self.ui_manager.set_status(self.ui.statusLabel, "已停止")
    
    def _detect_image(self, image_path: str):
        """检测图片"""
        if not self.detection_service:
            return
        
        self.current_image_path = image_path
        self.current_result = self.detection_service.detect(image_path)
        
        # 显示结果
        plotted_image = self.current_result.get_plotted_image()
        self.ui_manager.display_image(self.ui.imageLabel, plotted_image)
        
        # 更新表格
        self.ui_manager.update_result_table(self.ui.resultTable, self.current_result)
        
        # 更新检测信息
        self.ui_manager.update_detection_info(
            self.ui.detectInfoLabel,
            self.current_result.count,
            self.current_result.elapsed_time
        )
        
        # 显示分类指导
        guides = self.current_result.get_classification_guide()
        self.ui_manager.show_classification_guide(self.ui.guideLabel, guides)
        
        # 记录统计
        if self.current_result.has_detections:
            self.statistics_manager.add_record(self.current_result, image_path)
            self._update_statistics_display()
    
    def _start_video(self, source, is_camera: bool = False):
        """启动视频处理"""
        if not self.detection_service:
            return
        
        self.on_stop()  # 停止之前的
        
        self.video_thread = VideoThread(self.detection_service)
        self.video_thread.set_source(source, is_camera)
        self.video_thread.frame_ready.connect(self._on_video_frame)
        self.video_thread.finished_signal.connect(self._on_video_finished)
        self.video_thread.start()
        
        self.ui_manager.set_status(
            self.ui.statusLabel, 
            "摄像头运行中..." if is_camera else "视频播放中..."
        )
    
    def _on_video_frame(self, frame, result):
        """处理视频帧"""
        self.current_result = result
        self.ui_manager.display_image(self.ui.imageLabel, frame)
        self.ui_manager.update_result_table(self.ui.resultTable, result)
        self.ui_manager.update_detection_info(
            self.ui.detectInfoLabel, result.count, result.elapsed_time
        )
        guides = result.get_classification_guide()
        self.ui_manager.show_classification_guide(self.ui.guideLabel, guides)
    
    def _on_video_finished(self):
        """视频处理完成"""
        self.is_camera_running = False
        self.ui.CapBtn.setText("开启摄像头")
        self.ui_manager.set_status(self.ui.statusLabel, "视频播放完成")
    
    def _update_statistics_display(self):
        """更新统计显示"""
        stats = self.statistics_manager.get_today_statistics()
        self.ui_manager.update_statistics_display(stats)
    
    def on_export_statistics(self):
        """导出统计"""
        try:
            export_path = self.statistics_manager.export_to_csv()
            QMessageBox.information(self, "提示", f"已导出到: {export_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {e}")
    
    def on_clear_statistics(self):
        """清空统计"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有统计记录吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.statistics_manager.clear_records()
            self._update_statistics_display()
            QMessageBox.information(self, "提示", "统计记录已清空")
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Escape:
            self.on_stop()
        elif event.key() == Qt.Key_Left and self.image_list:
            self._prev_image()
        elif event.key() == Qt.Key_Right and self.image_list:
            self._next_image()
    
    def _prev_image(self):
        """上一张图片"""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self._detect_image(self.image_list[self.current_image_index])
            self._update_image_status()
    
    def _next_image(self):
        """下一张图片"""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._detect_image(self.image_list[self.current_image_index])
            self._update_image_status()
    
    def _update_image_status(self):
        """更新图片状态"""
        self.ui_manager.set_status(
            self.ui.statusLabel,
            f"图片 {self.current_image_index + 1}/{len(self.image_list)}"
        )
    
    def closeEvent(self, event):
        """关闭事件"""
        self.on_stop()
        event.accept()
    
    def _update_config_menu_state(self):
        """更新配置菜单选中状态"""
        if not hasattr(self.ui, 'actionConfig5'):
            return
        
        current_mode = settings.CURRENT_MODE
        self.ui.actionConfig5.setChecked(current_mode == 'cls5')
        self.ui.actionConfig23.setChecked(current_mode == 'cls23')
        self.ui.actionConfig40.setChecked(current_mode == 'cls40')
        
        # 更新窗口标题显示当前配置
        config = settings.get_current_config()
        self.setWindowTitle(f"基于YOLOv8的垃圾目标检测系统 - {config['NUM_CLASSES']}类")
    
    def on_switch_config(self, mode: str):
        """切换配置模式"""
        # 停止当前视频/摄像头
        self.on_stop()
        
        # 获取配置信息
        config_names = {
            'cls5': ('5类配置', settings.MODEL_5CLS_PATH),
            'cls23': ('23类配置', settings.MODEL_23CLS_PATH),
            'cls40': ('40类配置', settings.MODEL_40CLS_PATH),
        }
        config_name, model_path = config_names.get(mode, ('未知', None))
        
        # 检查模型是否存在
        if model_path and not Path(model_path).exists():
            reply = QMessageBox.warning(
                self, "模型不存在",
                f"未找到 {config_name} 的模型文件:\n{model_path}\n\n"
                "是否仍然切换配置？（将使用默认模型）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self._update_config_menu_state()
                return
        
        # 切换配置
        settings.set_mode(mode)
        
        # 重新加载模型
        try:
            model_path_str = str(settings.CURRENT_MODEL_PATH)
            if Path(model_path_str).exists():
                self.detection_service = DetectionService(model_path_str)
                self.ui_manager.set_status(
                    self.ui.statusLabel,
                    f"已切换到 {config_name}，模型加载成功"
                )
            else:
                # 模型不存在，仅切换配置
                self.detection_service = None
                self.ui_manager.set_status(
                    self.ui.statusLabel,
                    f"已切换到 {config_name}，等待模型文件"
                )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
            self.detection_service = None
        
        # 更新菜单状态
        self._update_config_menu_state()
        
        # 提示用户
        config = settings.get_current_config()
        QMessageBox.information(
            self, "配置已切换",
            f"已切换到 {config_name}\n"
            f"类别数量: {config['NUM_CLASSES']}\n"
            f"模型路径: {config['model_path']}"
        )


def main():
    """主函数"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    app = QApplication(sys.argv)
    app.setApplicationName("垃圾目标检测系统")
    
    window = MainWindow()
    window.setWindowTitle("基于YOLOv8的垃圾目标检测系统")
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
