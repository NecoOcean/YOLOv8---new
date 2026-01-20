# UiMain.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 主窗口UI定义
"""
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 700))
        
        # 设置窗口图标和标题
        MainWindow.setWindowTitle("基于YOLOv8的垃圾目标检测系统")
        
        # 中央部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 主布局
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.mainLayout.setSpacing(10)
        
        # 左侧面板 - 图像显示区域
        self.leftPanel = QtWidgets.QWidget()
        self.leftLayout = QtWidgets.QVBoxLayout(self.leftPanel)
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        
        # 图像显示标签
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setMinimumSize(QtCore.QSize(640, 480))
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setText("请选择图片、视频或开启摄像头")
        self.imageLabel.setStyleSheet("""
            QLabel {
                background-color: #ECEFF1;
                border: 2px dashed #90A4AE;
                border-radius: 8px;
                font-size: 16px;
                color: #607D8B;
            }
        """)
        self.leftLayout.addWidget(self.imageLabel)
        
        # 状态信息标签
        self.statusLabel = QtWidgets.QLabel("就绪")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setStyleSheet("font-size: 14px; color: #4CAF50; padding: 5px;")
        self.leftLayout.addWidget(self.statusLabel)
        
        self.mainLayout.addWidget(self.leftPanel, stretch=3)
        
        # 右侧面板 - 控制和结果区域
        self.rightPanel = QtWidgets.QWidget()
        self.rightPanel.setMaximumWidth(400)
        self.rightLayout = QtWidgets.QVBoxLayout(self.rightPanel)
        self.rightLayout.setContentsMargins(0, 0, 0, 0)
        self.rightLayout.setSpacing(10)
        
        # 控制按钮组
        self.controlGroup = QtWidgets.QGroupBox("操作控制")
        self.controlLayout = QtWidgets.QGridLayout(self.controlGroup)
        
        self.PicBtn = QtWidgets.QPushButton("📷 打开图片")
        self.PicBtn.setMinimumHeight(40)
        self.controlLayout.addWidget(self.PicBtn, 0, 0)
        
        self.FolderBtn = QtWidgets.QPushButton("📁 打开文件夹")
        self.FolderBtn.setMinimumHeight(40)
        self.controlLayout.addWidget(self.FolderBtn, 0, 1)
        
        self.VideoBtn = QtWidgets.QPushButton("🎬 打开视频")
        self.VideoBtn.setMinimumHeight(40)
        self.controlLayout.addWidget(self.VideoBtn, 1, 0)
        
        self.CapBtn = QtWidgets.QPushButton("📹 开启摄像头")
        self.CapBtn.setMinimumHeight(40)
        self.controlLayout.addWidget(self.CapBtn, 1, 1)
        
        self.SaveBtn = QtWidgets.QPushButton("💾 保存结果")
        self.SaveBtn.setMinimumHeight(40)
        self.controlLayout.addWidget(self.SaveBtn, 2, 0)
        
        self.StopBtn = QtWidgets.QPushButton("⏹ 停止")
        self.StopBtn.setMinimumHeight(40)
        self.StopBtn.setEnabled(False)
        self.controlLayout.addWidget(self.StopBtn, 2, 1)
        
        self.rightLayout.addWidget(self.controlGroup)
        
        # 检测信息组
        self.infoGroup = QtWidgets.QGroupBox("检测信息")
        self.infoLayout = QtWidgets.QVBoxLayout(self.infoGroup)
        
        self.detectInfoLabel = QtWidgets.QLabel("等待检测...")
        self.detectInfoLabel.setWordWrap(True)
        self.detectInfoLabel.setStyleSheet("font-size: 13px; padding: 5px;")
        self.infoLayout.addWidget(self.detectInfoLabel)
        
        self.rightLayout.addWidget(self.infoGroup)
        
        # 检测结果表格
        self.resultGroup = QtWidgets.QGroupBox("检测结果")
        self.resultLayout = QtWidgets.QVBoxLayout(self.resultGroup)
        
        self.resultTable = QtWidgets.QTableWidget()
        self.resultTable.setColumnCount(4)
        self.resultTable.setHorizontalHeaderLabels(["类别", "置信度", "位置", "分类"])
        self.resultTable.horizontalHeader().setStretchLastSection(True)
        self.resultTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.resultTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.resultTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.resultLayout.addWidget(self.resultTable)
        
        self.rightLayout.addWidget(self.resultGroup, stretch=1)
        
        # 分类指导组
        self.guideGroup = QtWidgets.QGroupBox("分类指导")
        self.guideLayout = QtWidgets.QVBoxLayout(self.guideGroup)
        
        self.guideLabel = QtWidgets.QLabel("暂无分类指导")
        self.guideLabel.setWordWrap(True)
        self.guideLabel.setStyleSheet("""
            font-size: 13px;
            padding: 10px;
            background-color: #E8F5E9;
            border-radius: 4px;
            line-height: 1.5;
        """)
        self.guideLayout.addWidget(self.guideLabel)
        
        self.rightLayout.addWidget(self.guideGroup)
        
        # 统计面板组
        self.statsGroup = QtWidgets.QGroupBox("检测统计")
        self.statsLayout = QtWidgets.QVBoxLayout(self.statsGroup)
        
        # 今日统计标签
        self.todayStatsLabel = QtWidgets.QLabel("今日检测: 0 次 | 共 0 项")
        self.todayStatsLabel.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")
        self.statsLayout.addWidget(self.todayStatsLabel)
        
        # 分类统计网格
        self.categoryStatsWidget = QtWidgets.QWidget()
        self.categoryStatsLayout = QtWidgets.QGridLayout(self.categoryStatsWidget)
        self.categoryStatsLayout.setSpacing(5)
        
        # 四大类统计标签
        self.kitchenWasteLabel = QtWidgets.QLabel("🟢 厨余垃圾: 0")
        self.kitchenWasteLabel.setStyleSheet("background-color: #E8F5E9; padding: 8px; border-radius: 4px;")
        self.categoryStatsLayout.addWidget(self.kitchenWasteLabel, 0, 0)
        
        self.recyclableLabel = QtWidgets.QLabel("🔵 可回收物: 0")
        self.recyclableLabel.setStyleSheet("background-color: #E3F2FD; padding: 8px; border-radius: 4px;")
        self.categoryStatsLayout.addWidget(self.recyclableLabel, 0, 1)
        
        self.hazardousLabel = QtWidgets.QLabel("🔴 有害垃圾: 0")
        self.hazardousLabel.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px;")
        self.categoryStatsLayout.addWidget(self.hazardousLabel, 1, 0)
        
        self.otherWasteLabel = QtWidgets.QLabel("⚫ 其他垃圾: 0")
        self.otherWasteLabel.setStyleSheet("background-color: #ECEFF1; padding: 8px; border-radius: 4px;")
        self.categoryStatsLayout.addWidget(self.otherWasteLabel, 1, 1)
        
        self.statsLayout.addWidget(self.categoryStatsWidget)
        
        # 统计操作按钮
        self.statsButtonLayout = QtWidgets.QHBoxLayout()
        
        self.exportStatsBtn = QtWidgets.QPushButton("📊 导出统计")
        self.exportStatsBtn.setMinimumHeight(30)
        self.statsButtonLayout.addWidget(self.exportStatsBtn)
        
        self.clearStatsBtn = QtWidgets.QPushButton("🗑 清空记录")
        self.clearStatsBtn.setMinimumHeight(30)
        self.statsButtonLayout.addWidget(self.clearStatsBtn)
        
        self.statsLayout.addLayout(self.statsButtonLayout)
        
        self.rightLayout.addWidget(self.statsGroup)
        
        self.mainLayout.addWidget(self.rightPanel, stretch=1)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 25))
        
        self.menuFile = QtWidgets.QMenu("文件", self.menubar)
        self.menuConfig = QtWidgets.QMenu("配置", self.menubar)
        self.menuHelp = QtWidgets.QMenu("帮助", self.menubar)
        
        MainWindow.setMenuBar(self.menubar)
        
        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        
        # 菜单动作 - 文件
        self.actionOpen = QtWidgets.QAction("打开图片", MainWindow)
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionExit = QtWidgets.QAction("退出", MainWindow)
        self.actionExit.setShortcut("Ctrl+Q")
        
        # 菜单动作 - 配置切换
        self.actionConfig5 = QtWidgets.QAction("5类配置 (简化版)", MainWindow)
        self.actionConfig5.setCheckable(True)
        self.actionConfig23 = QtWidgets.QAction("23类配置 (标准版)", MainWindow)
        self.actionConfig23.setCheckable(True)
        self.actionConfig40 = QtWidgets.QAction("40类配置 (精细版)", MainWindow)
        self.actionConfig40.setCheckable(True)
        
        # 配置互斥组
        self.configActionGroup = QtWidgets.QActionGroup(MainWindow)
        self.configActionGroup.addAction(self.actionConfig5)
        self.configActionGroup.addAction(self.actionConfig23)
        self.configActionGroup.addAction(self.actionConfig40)
        self.configActionGroup.setExclusive(True)
        
        # 菜单动作 - 帮助
        self.actionAbout = QtWidgets.QAction("关于", MainWindow)
        
        # 组装菜单
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        
        self.menuConfig.addAction(self.actionConfig5)
        self.menuConfig.addAction(self.actionConfig23)
        self.menuConfig.addAction(self.actionConfig40)
        
        self.menuHelp.addAction(self.actionAbout)
        
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuConfig.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """
        for btn in [self.PicBtn, self.FolderBtn, self.VideoBtn, self.CapBtn, self.SaveBtn, self.StopBtn]:
            btn.setStyleSheet(button_style)
        
        # 统计按钮使用不同样式
        stats_button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
        """
        self.exportStatsBtn.setStyleSheet(stats_button_style)
        
        clear_button_style = """
            QPushButton {
                background-color: #FF5722;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #BF360C;
            }
        """
        self.clearStatsBtn.setStyleSheet(clear_button_style)
