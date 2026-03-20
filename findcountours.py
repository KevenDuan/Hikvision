# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import ctypes

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

    # Check current Exposure Time
    # stFloatValue = MVCC_FLOATVALUE()
    # memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
    # ret = cam.MV_CC_GetFloatValue("ExposureTime", stFloatValue)
    # if ret == 0:
    #     print("Current ExposureTime: %f us" % stFloatValue.fCurValue)
    # else:
    #     print("Get ExposureTime failed! ret[0x%x]" % ret)
    
    # 设置曝光时间
    exposure_time = 5000.0
    ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
    if ret != 0:
        print("Set ExposureTime failed! ret[0x%x]" % ret)
    else:
        print(f"Set ExposureTime to {exposure_time} us")

    # 配置为连续取流模式 (关闭硬件触发模式)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing failed! ret[0x%x]" % ret)
        sys.exit()
        
    print("Successfully start grabbing! Press 'q' to exit.")

    # 创建用于接收底层图像数据的结构体
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    while True:
        # 从相机缓存中抓取一帧数据 (超时时间设置为 1000ms)
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        
        if ret == 0:
            # ---------------------------------------------------------
            # 核心桥接技术：将 C 语言内存指针强转为 OpenCV 矩阵
            # ---------------------------------------------------------
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            
            # 使用 ctypes 高效读取底层内存区块
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            
            # 将一维数据转为 np.uint8，重塑为二维矩阵，并执行深拷贝 (.copy()) 获取写入权限！
            img_array = np.frombuffer(data, dtype=np.uint8)
            cv_image = img_array.reshape((nHeight, nWidth)).copy()


            # ---------------------------------------------------------
            # OpenCV 算法处理区 (你的视觉测算逻辑全部写在这里)
            # ---------------------------------------------------------
            
            # 1. 转为彩色图像以便绘制彩色边框 (如果原图是单通道灰度图)
            if len(cv_image.shape) == 2:
                cv_image_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                cv_image_color = cv_image.copy()

            # 2. 高斯模糊去噪
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

            # 3. Canny 边缘检测 (阈值可根据实际光照调整)
            edges = cv2.Canny(blurred, 50, 150)

            # 4. 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 5. 遍历轮廓并绘制外接矩形 (Bounding Box)
            for contour in contours:
                # 过滤掉太小的噪点轮廓 (面积 < 100)
                if cv2.contourArea(contour) < 10000:
                    continue
                
                # 获取外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在彩色图上画出绿色矩形框 (BGR: 0, 255, 0)
                cv2.rectangle(cv_image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 在左上角实时显示检测到的轮廓数量
            cv2.putText(cv_image_color, f"Contours: {len(contours)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 调用 OpenCV 窗口显示最终图像
            # 增加 WINDOW_KEEPRATIO 保证画面不被拉伸变形
            cv2.namedWindow("Vision Test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            # 强制初始窗口大小为你电脑看起来最舒服的尺寸（比如 800x640，保持 5:4 比例）
            cv2.resizeWindow("Vision Test", 800, 640)
            cv2.imshow("Vision Test", cv_image_color)

            # 必须步骤：释放该帧内存，否则几秒钟后内存就会爆满导致程序崩溃
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        else:
            print(f"Get image failed! ret[0x%x]" % ret)

        # 监听键盘输入，按 'q' 键退出死循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 优雅地收尾销毁资源
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()
    print("Program exit successfully.")

if __name__ == "__main__":
    main()