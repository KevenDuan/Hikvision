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
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if deviceList.nDeviceNum == 0: sys.exit()
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice()
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetFloatValue("ExposureTime", 10000)
    cam.MV_CC_StartGrabbing()

    print("\n" + "="*50)
    print("【抗遮挡：霍夫圆满级重建系统】")
    print("请放入圆形零件，并用手遮挡大部分边缘！")
    print("按 'q' 键退出。")
    print("="*50 + "\n")

    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret == 0:
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            img_array = np.frombuffer(data, dtype=np.uint8)
            cv_image = img_array.reshape((nHeight, nWidth)).copy()
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            # 1. 极其关键的预处理：高斯模糊
            # 霍夫圆对噪点极其敏感，必须要用大卷积核把表面的杂质和微小反光抹平
            blurred = cv2.GaussianBlur(cv_image, (9, 9), 2)

            # 2. 核心数学武器：霍夫梯度法检测圆
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1,              # 累加器分辨率与图像分辨率的比值，默认1即可
                minDist=100,       # 画面中两个圆心之间允许的最短距离 (防止一个零件画出十几个圈)
                param1=100,        # 内部 Canny 边缘检测的高阈值 (调高可以过滤浅色杂纹)
                param2=50,         # 【最核心参数】中心点得票数阈值！越低越容易误判，越高要求圆越完美
                minRadius=20,      # 你要找的零件的最小像素半径
                maxRadius=400      # 最大像素半径
            )

            # 如果画面里存在符合数学逻辑的圆
            if circles is not None:
                # 提取数据，并按照得票数(置信度)自动降序排列
                circles = np.round(circles[0, :]).astype("int")
                
                # 我们只取画面中得票数最高、最像圆的那一个！
                for (x, y, r) in circles:
                    
                    # 3. 完美复原：画出虚拟重建的完整黄色边界！
                    cv2.circle(display_image, (x, y), r, (0, 255, 255), 3)

                    # 画出红色的绝对圆心准星
                    cv2.drawMarker(display_image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    
                    # 打印坐标和虚拟半径
                    cv2.putText(display_image, f"Center: ({x}, {y})", (x + 30, y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(display_image, f"Radius: {r} px", (x + 30, y + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # 只要最大的圆，画完直接 break 退出循环
                    break

            # 画中画：帮你观察高斯模糊后的图像，太糊或太清晰都不利于找圆
            small_blur = cv2.resize(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), (320, 240))
            display_image[0:240, display_image.shape[1]-320:display_image.shape[1]] = small_blur

            cv2.namedWindow("Hough Circle Fitter", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Hough Circle Fitter", 800, 640)
            cv2.imshow("Hough Circle Fitter", display_image)
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()