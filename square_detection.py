# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import ctypes
import math

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    print(" [error] Import MvCameraControl_class failed!")
    sys.exit()

def get_distance(p1, p2):
    """计算两点之间的欧几里得距离 (像素长度)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    # 初始化海康 SDK
    MvCamera.MV_CC_Initialize()
    
    # 枚举设备
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

    # 打开设备
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    ret = cam.MV_CC_OpenDevice()
    if ret != 0:
        print("Open device failed! ret[0x%x]" % ret)
        sys.exit()

    # 设置曝光时间
    exposure_time = 20000.0
    ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
    if ret != 0:
        print("Set ExposureTime failed! ret[0x%x]" % ret)
    else:
        print(f"Set ExposureTime to {exposure_time} us")

    # 连续取流模式
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing failed! ret[0x%x]" % ret)
        sys.exit()
        
    print("Successfully start grabbing! Press 'q' to exit.")

    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        
        if ret == 0:
            # 获取图像并转换
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            img_array = np.frombuffer(data, dtype=np.uint8)
            
            gray_frame = img_array.reshape((nHeight, nWidth)).copy()
            color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # ---------------------------------------------------------
            # 核心算法：基于轮廓的角点检测与实际测量
            # ---------------------------------------------------------
            
            # 标定系数：根据用户提供的数据 (468px/10.6cm, 650px/14.6cm) 计算平均值
            PIXELS_PER_CM = 44.33 

            # 1. 图像预处理：模糊去噪
            blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            
            # 2. 边缘检测
            edged = cv2.Canny(blurred, 50, 150)

            cv2.namedWindow("edge", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            cv2.resizeWindow("edge", 800, 640)
            cv2.imshow("edge", edged)
            
            # 3. 查找轮廓
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 过滤掉面积太小的噪点
                if cv2.contourArea(contour) < 10000: # 稍微调大面积阈值
                    continue
                
                # 4. 轮廓近似 (相当于角点检测)
                # 0.02 * peri 是拟合精度，值越大拟合越粗糙，越小越精确
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True) # 修改：0.02 --> 0.01
                
                # 只要是多边形 (角点数 >= 3)，就进行测量和绘制
                if len(approx) >= 3:
                    # 绘制多边形轮廓 (连接所有角点)
                    cv2.drawContours(color_frame, [approx], -1, (0, 255, 0), 2)
                    
                    # 获取所有角点
                    pts = approx.reshape(-1, 2)
                    num_points = len(pts)
                    
                    # 遍历每一条边，计算长度并在画面上标注
                    for i in range(num_points):
                        # 当前顶点 p1 和下一个顶点 p2
                        p1 = pts[i]
                        p2 = pts[(i + 1) % num_points]
                        
                        # 计算像素距离
                        pixel_dist = get_distance(p1, p2)
                        
                        # 转换为实际厘米
                        real_dist_cm = pixel_dist / PIXELS_PER_CM
                        
                        # 计算中点，用于放置文字
                        mid_x = (p1[0] + p2[0]) // 2
                        mid_y = (p1[1] + p2[1]) // 2
                        
                        # 在中点处绘制实际长度 (红色文字)
                        cv2.putText(color_frame, f"{real_dist_cm:.1f}cm", (mid_x, mid_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # 绘制角点 (黄色小圆点)
                        cv2.circle(color_frame, tuple(p1), 5, (0, 255, 255), -1)
                        # 可以在角点旁标出坐标
                        # cv2.putText(color_frame, f"{p1}", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 在左上角显示提示
            cv2.putText(color_frame, f"Scale: {PIXELS_PER_CM:.2f} px/cm", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 显示结果
            cv2.namedWindow("Measurement", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            cv2.resizeWindow("Measurement", 800, 640)
            cv2.imshow("Measurement", color_frame)

            cam.MV_CC_FreeImageBuffer(stOutFrame)

        else:
            print(f"Get image failed! ret[0x%x]" % ret)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()
    print("Program exit successfully.")

if __name__ == "__main__":
    main()
