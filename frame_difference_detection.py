# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import ctypes
import time

# 导入海康 SDK 核心库 (确保 MvImport 文件夹与本脚本同级)
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    print(" [error] Import MvCameraControl_class failed!")
    sys.exit()

def main():
    # 初始化海康 SDK
    MvCamera.MV_CC_Initialize()
    
    # 枚举局域网内的相机设备 (支持 GigE 和 USB 相机)
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("Find device failed! ret[0x%x]" % ret)
        sys.exit()
    if deviceList.nDeviceNum == 0:
        print("No device found!")
        sys.exit()

    print(f"find {deviceList.nDeviceNum} devices, link first device.")

    # 创建相机实例并打开设备
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    ret = cam.MV_CC_OpenDevice()
    if ret != 0:
        print("Open device failed! ret[0x%x]" % ret)
        sys.exit()

    # 配置为连续取流模式 (关闭硬件触发模式)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing failed! ret[0x%x]" % ret)
        sys.exit()
        
    print("Successfully start grabbing!")
    print("Instructions:")
    print("  Press 'b' to capture/update background frame (ensure machine is empty).")
    print("  Press 'q' to exit.")

    # 创建用于接收底层图像数据的结构体
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    # 初始化背景帧变量
    background_gray = None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 形态学操作核

    while True:
        # 从相机缓存中抓取一帧数据 (超时时间设置为 1000ms)
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        
        if ret == 0:
            # ---------------------------------------------------------
            # 图像获取与转换
            # ---------------------------------------------------------
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            img_array = np.frombuffer(data, dtype=np.uint8)
            
            # 此时得到的可能是单通道(灰度)或 Bayer 格式，为了通用性，先转为 BGR 彩色显示
            # 注意：如果相机本身输出的是 Mono8，reshape 后就是灰度图
            if stOutFrame.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                current_frame_gray = img_array.reshape((nHeight, nWidth)).copy()
                current_frame_color = cv2.cvtColor(current_frame_gray, cv2.COLOR_GRAY2BGR)
            else:
                # 如果是彩色相机，这里简化处理，假设是 Mono8 以演示逻辑。
                # 实际如果是 Bayer 格式需要 MV_CC_ConvertPixelType，这里暂且按 Mono8 处理
                # 如果遇到报错，请确认相机像素格式设置
                current_frame_gray = img_array.reshape((nHeight, nWidth)).copy() 
                current_frame_color = cv2.cvtColor(current_frame_gray, cv2.COLOR_GRAY2BGR)

            # 对当前帧进行高斯模糊，减少噪点影响
            blurred_frame = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)

            # ---------------------------------------------------------
            # 核心算法：帧差法 (Background Subtraction)
            # ---------------------------------------------------------
            if background_gray is not None:
                # 1. 计算当前帧与背景帧的绝对差
                frame_diff = cv2.absdiff(background_gray, blurred_frame)

                # 2. 二值化处理 (阈值 30 可根据环境光照调整)
                _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

                # 3. 形态学操作：开运算 (先腐蚀后膨胀) 去除微小噪点
                # 这一步对于“环境乱”非常重要，能滤掉切屑反光等噪点
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                # 膨胀操作，让目标物体区域连接得更紧密
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

                # 4. 查找轮廓
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 5. 遍历轮廓并绘制
                count = 0
                for contour in contours:
                    # 过滤掉面积过小的噪点 (例如小于 500 像素)
                    if cv2.contourArea(contour) < 500:
                        continue
                    
                    count += 1
                    # 获取外接矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 在彩色图上画出绿色矩形框
                    cv2.rectangle(current_frame_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 标出中心点坐标
                    cx, cy = x + w//2, y + h//2
                    cv2.circle(current_frame_color, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(current_frame_color, f"({cx},{cy})", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(current_frame_color, f"Detected: {count}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 显示二值化后的差异图 (调试用，方便看是否噪点太多)
                cv2.namedWindow("Diff Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow("Diff Mask", 400, 300)
                cv2.imshow("Diff Mask", thresh)

            else:
                # 如果还没有背景，提示用户按下 'b'
                cv2.putText(current_frame_color, "Press 'b' to capture background", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ---------------------------------------------------------
            # 显示控制
            # ---------------------------------------------------------
            cv2.namedWindow("Frame Difference Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            cv2.resizeWindow("Frame Difference Detection", 800, 640)
            cv2.imshow("Frame Difference Detection", current_frame_color)

            # 释放海康帧缓存
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        else:
            print(f"Get image failed! ret[0x%x]" % ret)

        # ---------------------------------------------------------
        # 键盘交互
        # ---------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # 按下 'b' 键，捕获当前帧作为背景
            # 注意：必须使用高斯模糊后的灰度图作为背景
            if 'blurred_frame' in locals():
                background_gray = blurred_frame.copy()
                print("Background captured successfully!")

    # 优雅地收尾销毁资源
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()
    print("Program exit successfully.")

if __name__ == "__main__":
    main()
