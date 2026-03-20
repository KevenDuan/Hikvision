# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import ctypes

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    print("【错误】找不到 MvImport 库！")
    sys.exit()

def main():
    # =========================================================
    # 🎯 物理世界核心参数配置区 (请根据实际情况修改)
    # =========================================================
    H_cam = 500.0           # 相机镜头到机床桌面的绝对物理高度 (毫米)
    Ratio_table = 0.225051  # 你刚刚用A4纸标定出来的桌面比例尺 (毫米/像素)
    h_part = 14            # 测试零件的高度 (毫米)。如果是平贴在桌面的卡片或钢尺，请填 0.0
    # =========================================================

    # 1. 初始化并连接相机
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    
    if deviceList.nDeviceNum == 0:
        print("未找到相机！")
        sys.exit()

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice()
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_StartGrabbing()
        
    print("成功开启视频流！正在进行精密尺寸测量... 按 'q' 键退出。")

    # 设置曝光时间
    cam.MV_CC_SetFloatValue("ExposureTime", 20000)

    # 计算补偿后的真实比例尺 (相似三角形原理)
    Ratio_real = Ratio_table * (1 - h_part / H_cam)

    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        
        if ret == 0:
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            
            # 读取内存并强转为 OpenCV 矩阵
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            img_array = np.frombuffer(data, dtype=np.uint8)
            cv_image = img_array.reshape((nHeight, nWidth)).copy()
            
            # 转为彩色画布
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            # --- 图像预处理与轮廓提取 ---
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            # 使用 OTSU 算法自动二值化 (白物黑底)
            ret_thresh, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- 遍历轮廓并测算尺寸 ---
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # 过滤掉小于 1500 像素的灰尘和反光点
                if area > 1500: 
                    # 获取带旋转角度的最小外接矩形
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w_pixel, h_pixel), angle = rect

                    # 【核心运算】将像素长宽乘以修正后的真实比例尺，得出物理毫米数
                    w_mm = w_pixel * Ratio_real
                    h_mm = h_pixel * Ratio_real

                    # 区分长边和宽边
                    length_mm = max(w_mm, h_mm)
                    width_mm = min(w_mm, h_mm)

                    # 画出蓝色外接矩形框
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)  # 坐标必须转为整数
                    cv2.drawContours(display_image, [box], 0, (255, 0, 0), 2)

                    # 在画面上实时打印数据
                    # 1. 打印像素中心坐标 (红字)
                    cv2.putText(display_image, f"Center: ({int(cx)}, {int(cy)})", (int(cx) + 20, int(cy) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 2. 打印物理长宽毫米数 (蓝字)
                    cv2.putText(display_image, f"L: {length_mm:.2f} mm", (int(cx) + 20, int(cy) + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(display_image, f"W: {width_mm:.2f} mm", (int(cx) + 20, int(cy) + 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 画中心固定十字准星
            center_x, center_y = nWidth // 2, nHeight // 2
            cv2.line(display_image, (center_x - 50, center_y), (center_x + 50, center_y), (255, 255, 255), 2)
            cv2.line(display_image, (center_x, center_y - 50), (center_x, center_y + 50), (255, 255, 255), 2)
            
            # 显示画面
            cv2.namedWindow("CNC Vision - Dimension Measurement", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            cv2.resizeWindow("CNC Vision - Dimension Measurement", 800, 640)
            cv2.imshow("CNC Vision - Dimension Measurement", display_image)
            
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 优雅收尾
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()
    print("程序已安全退出。")

if __name__ == "__main__":
    main()