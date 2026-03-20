# Hikvision海康相机开发

## 海康官方库

确保 `MvImport` 文件夹与本脚本同级，且包含所有必要的 `.py` 文件。

> `MvImport\MvCameraControl_class.py` 官方相机控制类    
> `MvImport\CameraParams_const.py` 相机参数常量   
> `MvImport\CameraParams_header.py` 相机参数头文件  
> `MvImport\MvErrorDefine_const.py` 相机错误码常量  
> `MvImport\MvISPErrorDefine_const.py` 相机ISP错误码常量  
> `MvImport\PixelType_header.py` 像素类型头文件  

## 相机初始化

详细参考`demo_hk2cv.py` -> 从海康读取数据流并转换成numpy数组给OpenCV使用

```python
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
    # 枚举 GigE 相机和 USB 相机
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    # 枚举相机设备
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
    # 强制转换为 MV_CC_DEVICE_INFO 结构体指针
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    # 创建相机句柄
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    # 打开相机设备
    ret = cam.MV_CC_OpenDevice()
    if ret != 0:
        print("Open device failed! ret[0x%x]" % ret)
        sys.exit()

    # 查看当前曝光时间
    # stFloatValue = MVCC_FLOATVALUE()
    # memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
    # ret = cam.MV_CC_GetFloatValue("ExposureTime", stFloatValue)
    # if ret == 0:
    #     print("Current ExposureTime: %f us" % stFloatValue.fCurValue)
    # else:
    #     print("Get ExposureTime failed! ret[0x%x]" % ret)
    
    # 设置曝光时间
    exposure_time = 10000.0
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

            # 在画面正中心画一个十字准星
            center_x, center_y = nWidth // 2, nHeight // 2
            cv2.line(cv_image, (center_x - 50, center_y), (center_x + 50, center_y), (255, 255, 255), 2)
            cv2.line(cv_image, (center_x, center_y - 50), (center_x, center_y + 50), (255, 255, 255), 2)
            
            # 在左上角实时显示分辨率信息
            cv2.putText(cv_image, f"Resolution: {nWidth} x {nHeight}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 调用 OpenCV 窗口显示最终图像
            # 增加 WINDOW_KEEPRATIO 保证画面不被拉伸变形
            cv2.namedWindow("Vision Test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            # 强制初始窗口大小为你电脑看起来最舒服的尺寸（比如 800x640，保持 5:4 比例）
            cv2.resizeWindow("Vision Test", 800, 640)
            cv2.imshow("Vision Test", cv_image)

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
```

## 相机内参数校准

使用标定板放在相机视野内，运行`calibrate_ratio.py`程序，程序会自动检测标定板角点并计算内参。

注意：  
`Square Size` 为标定板的单格边长，单位为毫米。  
`Checkerboard` 为棋盘格内角点数量，格式为 (横向角点数, 纵向角点数)。  

具体代码如下：
```python
SQUARE_SIZE_MM = 28.6  # GP200-5 标定板的单格边长 毫米
CHECKERBOARD = (9, 6)  # 棋盘格内角点数量 (横向角点数, 纵向角点数)，请根据实际画面视野修改！
                       # 注意：角点数 = 黑白方块数 - 1
```