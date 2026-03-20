import sys
import ctypes

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    raise RuntimeError("找不到 MvImport（MvCameraControl_class.py）")


class MeasureWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    status = pyqtSignal(str)
    measurement = pyqtSignal(object)

    def __init__(self, h_cam_mm, ratio_table_mm_per_px, h_part_mm, exposure_us, area_min_px2, parent=None):
        super().__init__(parent)
        self._stop = False
        self._h_cam_mm = float(h_cam_mm)
        self._ratio_table = float(ratio_table_mm_per_px)
        self._h_part_mm = float(h_part_mm)
        self._exposure_us = float(exposure_us)
        self._area_min = float(area_min_px2)
        self._cam = None

    def stop(self):
        self._stop = True

    def run(self):
        try:
            MvCamera.MV_CC_Initialize()

            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
            if ret != 0:
                self.status.emit(f"枚举设备失败：0x{ret:x}")
                return
            if device_list.nDeviceNum == 0:
                self.status.emit("未找到相机")
                return

            cam = MvCamera()
            st_device = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = cam.MV_CC_CreateHandle(st_device)
            if ret != 0:
                self.status.emit(f"CreateHandle 失败：0x{ret:x}")
                return
            ret = cam.MV_CC_OpenDevice()
            if ret != 0:
                self.status.emit(f"OpenDevice 失败：0x{ret:x}")
                return

            self._cam = cam

            cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            cam.MV_CC_SetFloatValue("ExposureTime", self._exposure_us)

            ratio_real = self._ratio_table * (1.0 - self._h_part_mm / self._h_cam_mm)

            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.status.emit(f"StartGrabbing 失败：0x{ret:x}")
                return

            self.status.emit("采集中，按停止结束")

            st_out = MV_FRAME_OUT()
            memset(byref(st_out), 0, sizeof(st_out))

            while not self._stop:
                ret = cam.MV_CC_GetImageBuffer(st_out, 1000)
                if ret != 0:
                    continue

                n_height = st_out.stFrameInfo.nHeight
                n_width = st_out.stFrameInfo.nWidth
                n_frame_len = st_out.stFrameInfo.nFrameLen

                data = ctypes.string_at(st_out.pBufAddr, n_frame_len)
                img_array = np.frombuffer(data, dtype=np.uint8)
                cv_image = img_array.reshape((n_height, n_width)).copy()

                display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

                blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                best = None
                best_area = 0.0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area <= self._area_min:
                        continue
                    if area > best_area:
                        best_area = area

                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w_px, h_px), _ = rect

                    w_mm = w_px * ratio_real
                    h_mm = h_px * ratio_real
                    length_mm = max(w_mm, h_mm)
                    width_mm = min(w_mm, h_mm)
                    if area >= best_area:
                        best = {
                            "cx": float(cx),
                            "cy": float(cy),
                            "length_mm": float(length_mm),
                            "width_mm": float(width_mm),
                            "area_px2": float(area),
                        }

                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(display_image, [box], 0, (255, 0, 0), 2)

                    cv2.putText(
                        display_image,
                        f"Center: ({int(cx)}, {int(cy)})",
                        (int(cx) + 20, int(cy) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        display_image,
                        f"L: {length_mm:.2f} mm",
                        (int(cx) + 20, int(cy) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        display_image,
                        f"W: {width_mm:.2f} mm",
                        (int(cx) + 20, int(cy) + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                center_x, center_y = n_width // 2, n_height // 2
                cv2.line(display_image, (center_x - 50, center_y), (center_x + 50, center_y), (255, 255, 255), 2)
                cv2.line(display_image, (center_x, center_y - 50), (center_x, center_y + 50), (255, 255, 255), 2)

                rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                self.measurement.emit(best)

                cam.MV_CC_FreeImageBuffer(st_out)

        except Exception as e:
            self.status.emit(str(e))
        finally:
            try:
                if self._cam is not None:
                    try:
                        self._cam.MV_CC_StopGrabbing()
                    except Exception:
                        pass
                    try:
                        self._cam.MV_CC_CloseDevice()
                    except Exception:
                        pass
                    try:
                        self._cam.MV_CC_DestroyHandle()
                    except Exception:
                        pass
            finally:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                try:
                    MvCamera.MV_CC_Finalize()
                except Exception:
                    pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNC Vision - Dimension Measurement")
        self._worker = None
        self.resize(1600, 900)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1100, 800)

        self.h_cam = QDoubleSpinBox()
        self.h_cam.setRange(1.0, 100000.0)
        self.h_cam.setDecimals(1)
        self.h_cam.setValue(500.0)
        self.h_cam.setSuffix(" mm")

        self.ratio_table = QDoubleSpinBox()
        self.ratio_table.setRange(0.000001, 1000.0)
        self.ratio_table.setDecimals(6)
        self.ratio_table.setValue(0.225051)
        self.ratio_table.setSuffix(" mm/px")

        self.h_part = QDoubleSpinBox()
        self.h_part.setRange(0.0, 100000.0)
        self.h_part.setDecimals(1)
        self.h_part.setValue(14.0)
        self.h_part.setSuffix(" mm")

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(1.0, 10000000.0)
        self.exposure.setDecimals(0)
        self.exposure.setValue(20000.0)
        self.exposure.setSuffix(" us")

        self.area_min = QSpinBox()
        self.area_min.setRange(0, 100000000)
        self.area_min.setValue(1500)
        self.area_min.setSuffix(" px^2")

        form = QFormLayout()
        form.addRow("H_cam（相机高度）", self.h_cam)
        form.addRow("Ratio_table（mm/px）", self.ratio_table)
        form.addRow("h_part（零件高度）", self.h_part)
        form.addRow("ExposureTime", self.exposure)
        form.addRow("最小轮廓面积", self.area_min)

        params_box = QGroupBox("参数")
        params_box.setLayout(form)

        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        self.output.setMaximumBlockCount(200)
        self.output.appendPlainText("就绪")

        self.measure_box = QGroupBox("测量结果")
        self.measure_center = QLabel("Center: -")
        self.measure_l = QLabel("L: - mm")
        self.measure_w = QLabel("W: - mm")
        measure_layout = QVBoxLayout()
        measure_layout.addWidget(self.measure_center)
        measure_layout.addWidget(self.measure_l)
        measure_layout.addWidget(self.measure_w)
        self.measure_box.setLayout(measure_layout)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(params_box)
        left.addLayout(btn_row)
        left.addWidget(self.output)
        left.addWidget(self.measure_box)
        left.addStretch(1)

        main = QHBoxLayout()
        main.addLayout(left, 0)
        main.addWidget(self.image_label, 1)

        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)

    def start_worker(self):
        if self._worker is not None and self._worker.isRunning():
            return

        self._worker = MeasureWorker(
            self.h_cam.value(),
            self.ratio_table.value(),
            self.h_part.value(),
            self.exposure.value(),
            self.area_min.value(),
            self,
        )
        self._worker.frame_ready.connect(self.update_image)
        self._worker.status.connect(self.update_status)
        self._worker.measurement.connect(self.update_measurement)
        self._worker.finished.connect(self.on_finished)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def stop_worker(self):
        if self._worker is None:
            return
        self._worker.stop()
        self.btn_stop.setEnabled(False)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.output.appendPlainText("已停止")

    def update_status(self, text):
        if text:
            self.output.appendPlainText(text)

    def update_image(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_measurement(self, data):
        if not data:
            self.measure_center.setText("Center: -")
            self.measure_l.setText("L: - mm")
            self.measure_w.setText("W: - mm")
            return
        self.measure_center.setText(f"Center: ({data['cx']:.0f}, {data['cy']:.0f})")
        self.measure_l.setText(f"L: {data['length_mm']:.2f} mm")
        self.measure_w.setText(f"W: {data['width_mm']:.2f} mm")

    def closeEvent(self, event):
        self.stop_worker()
        if self._worker is not None:
            self._worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        QMessageBox.critical(None, "错误", str(e))
