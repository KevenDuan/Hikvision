# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import ctypes
import joblib
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    print("【错误】找不到 MvImport 库！请确保在正确目录下运行。")
    sys.exit()

# ==============================================================================
# 子线程：负责相机取流、按需开启 OpenCV 处理和 AI 推理
# ==============================================================================
class CameraThread(QThread):
    update_signal = pyqtSignal(QImage, float)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.is_measuring = False  # 【新增】测量状态控制开关
        self.cam = None
        
        self.cad_length = 51.0
        self.cad_width = 51.0
        
        self.IMG_W, self.IMG_H = 1280, 960
        self.FEATURE_COLS = ['norm_cx', 'norm_cy', 'long_px', 'short_px', 'cad_ratio', 'area_px']
        
        self.model_file = r'C:\Users\27732\Desktop\vision\MachineLearning\visual_height_model.pkl'
        try:
            self.ai_model = joblib.load(self.model_file)
            print("AI 模型加载成功！")
        except FileNotFoundError:
            self.ai_model = None
            print(f"【严重错误】找不到模型文件：{self.model_file}")

    def update_cad_params(self, length, width):
        self.cad_length = length
        self.cad_width = width

    def set_measuring_state(self, state):
        """控制是否进行 AI 推理与画框"""
        self.is_measuring = state

    def run(self):
        if self.ai_model is None:
            self.error_signal.emit("AI 模型未加载，无法启动！")
            return

        MvCamera.MV_CC_Initialize()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        
        if deviceList.nDeviceNum == 0:
            self.error_signal.emit("未检测到海康相机，请检查连接！")
            return
            
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam.MV_CC_CreateHandle(stDeviceList)
        self.cam.MV_CC_OpenDevice()
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_StartGrabbing()

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        
        self.is_running = True

        while self.is_running:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                nHeight = stOutFrame.stFrameInfo.nHeight
                nWidth = stOutFrame.stFrameInfo.nWidth
                nFrameLen = stOutFrame.stFrameInfo.nFrameLen
                
                data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
                img_array = np.frombuffer(data, dtype=np.uint8)
                cv_image = img_array.reshape((nHeight, nWidth)).copy()
                
                display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                current_predicted_height = -1.0 # -1 表示未测量

                # 【核心逻辑变更】只有按下测量按钮，才执行耗时的图像处理
                if self.is_measuring:
                    blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 3000:
                            rect = cv2.minAreaRect(cnt)
                            (cx, cy), (w_pixel, h_pixel), angle = rect
                            
                            long_px = max(w_pixel, h_pixel)
                            short_px = min(w_pixel, h_pixel)
                            
                            if short_px > 0 and self.cad_width > 0:
                                current_cad_ratio = self.cad_length / self.cad_width 
                                
                                features_dict = {
                                    'norm_cx': cx / self.IMG_W,
                                    'norm_cy': cy / self.IMG_H,
                                    'long_px': long_px,
                                    'short_px': short_px,
                                    'cad_ratio': current_cad_ratio,
                                    'area_px': area
                                }
                                features_df = pd.DataFrame([features_dict])[self.FEATURE_COLS]
                                
                                current_predicted_height = self.ai_model.predict(features_df)[0]

                                box = np.int32(cv2.boxPoints(rect))
                                cv2.drawContours(display_image, [box], 0, (255, 0, 0), 3) 
                                
                                text_h = f"Z: {current_predicted_height:.2f} mm"
                                cv2.putText(display_image, text_h, (int(cx) - 80, int(cy) - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                
                                break # 只处理最大的目标

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                self.update_signal.emit(qt_img, current_predicted_height)

        # 安全退出清理 (while 循环彻底结束后才执行)
        if self.cam is not None:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            MvCamera.MV_CC_Finalize()
            print("底层相机资源已安全释放。")

    def stop(self):
        """【修复死机BUG】只改变标志位，绝不使用 wait() 阻塞主线程"""
        self.is_running = False
        self.is_measuring = False


# ==============================================================================
# 主界面
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_thread = CameraThread()
        self.is_measuring_state = False # UI 层的测量状态机
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle("单目视觉 3D 深度测量系统")
        self.resize(1200, 800)
        
        main_layout = QHBoxLayout()
        
        self.video_label = QLabel("正在等待相机连接...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        self.video_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.video_label, stretch=3)
        
        control_layout = QVBoxLayout()
        
        # 1. 结果显示面板
        result_group = QGroupBox("实时测算结果")
        result_layout = QVBoxLayout()
        self.height_label = QLabel("---")
        self.height_label.setAlignment(Qt.AlignCenter)
        self.height_label.setStyleSheet("color: red; font-size: 48px; font-weight: bold;") 
        result_layout.addWidget(self.height_label)
        result_group.setLayout(result_layout)
        control_layout.addWidget(result_group)
        
        # 2. CAD 参数输入面板
        param_group = QGroupBox("当前零件物理尺寸 (CAD)")
        form_layout = QFormLayout()
        
        self.spin_length = QDoubleSpinBox()
        self.spin_length.setRange(1.0, 500.0)
        self.spin_length.setValue(51.0)
        self.spin_length.setSuffix(" mm")
        
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(1.0, 500.0)
        self.spin_width.setValue(51.0)
        self.spin_width.setSuffix(" mm")
        
        form_layout.addRow("零件长度 (长边):", self.spin_length)
        form_layout.addRow("零件宽度 (短边):", self.spin_width)
        param_group.setLayout(form_layout)
        control_layout.addWidget(param_group)
        
        # 3. 控制按钮
        # 相机连接组
        cam_layout = QHBoxLayout()
        self.btn_start_cam = QPushButton("打开相机")
        self.btn_start_cam.setMinimumHeight(40)
        self.btn_stop_cam = QPushButton("停止并断开")
        self.btn_stop_cam.setMinimumHeight(40)
        self.btn_stop_cam.setEnabled(False)
        cam_layout.addWidget(self.btn_start_cam)
        cam_layout.addWidget(self.btn_stop_cam)
        control_layout.addLayout(cam_layout)

        # 测量大按钮
        self.btn_measure = QPushButton("开始测量")
        self.btn_measure.setMinimumHeight(60)
        # 初始状态：绿色，被禁用（未连相机不能测）
        self.btn_measure.setStyleSheet("background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold;")
        self.btn_measure.setEnabled(False) 
        control_layout.addWidget(self.btn_measure)
        
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout, stretch=1)
        self.setLayout(main_layout)

    def connect_signals(self):
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_measure.clicked.connect(self.toggle_measure)
        
        self.spin_length.valueChanged.connect(self.update_thread_params)
        self.spin_width.valueChanged.connect(self.update_thread_params)
        
        self.camera_thread.update_signal.connect(self.update_frame)
        self.camera_thread.error_signal.connect(self.show_error)

    def update_thread_params(self):
        l = self.spin_length.value()
        w = self.spin_width.value()
        self.camera_thread.update_cad_params(l, w)

    def start_camera(self):
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self.btn_measure.setEnabled(True) # 相机开了，允许测算
        self.update_thread_params()
        self.camera_thread.start()
        self.video_label.setText("正在获取视频流...")

    def stop_camera(self):
        # 强行退出测量状态
        if self.is_measuring_state:
            self.toggle_measure()
            
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.btn_measure.setEnabled(False)
        
        # 安全下发停止指令
        self.camera_thread.stop()
        self.video_label.setText("相机已断开")
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        self.height_label.setText("---")

    def toggle_measure(self):
        """核心交互逻辑：切换测量状态，管控参数框的锁定"""
        if not self.is_measuring_state:
            # 动作：开始测量
            self.is_measuring_state = True
            self.btn_measure.setText("停止测量")
            # 变成红色警示状态
            self.btn_measure.setStyleSheet("background-color: #f44336; color: white; font-size: 20px; font-weight: bold;")
            
            # 【锁定参数输入】
            self.spin_length.setEnabled(False)
            self.spin_width.setEnabled(False)
            self.btn_stop_cam.setEnabled(False) # 测量时不准断开相机
            
            self.camera_thread.set_measuring_state(True)
        else:
            # 动作：停止测量
            self.is_measuring_state = False
            self.btn_measure.setText("开始测量")
            # 恢复绿色
            self.btn_measure.setStyleSheet("background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold;")
            
            # 【解锁参数输入】
            self.spin_length.setEnabled(True)
            self.spin_width.setEnabled(True)
            self.btn_stop_cam.setEnabled(True)
            
            self.camera_thread.set_measuring_state(False)
            self.height_label.setText("---") # 清空上一次的测量结果

    def update_frame(self, qt_img, current_height):
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        if self.is_measuring_state:
            if current_height >= 0:
                self.height_label.setText(f"{current_height:.2f} mm")
            else:
                self.height_label.setText("寻找目标...")

    def show_error(self, msg):
        self.video_label.setText(msg)
        self.video_label.setStyleSheet("background-color: black; color: red; font-size: 20px;")
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.btn_measure.setEnabled(False)

    def closeEvent(self, event):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait() # 只有在关闭整个软件时才允许 wait，保证底层资源不泄露
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())