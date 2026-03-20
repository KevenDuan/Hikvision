# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import ctypes
import math

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    print("【错误】找不到 MvImport 库！")
    sys.exit()

def main():
    # --- 1. 你买的标定板的绝对物理参数 ---
    SQUARE_SIZE_MM = 28.6  # GP200-5 标定板的单格边长：5.00 毫米
    CHECKERBOARD = (9, 6) # 棋盘格内角点数量 (横向角点数, 纵向角点数)，请根据实际画面视野修改！
                          # 注意：角点数 = 黑白方块数 - 1

    # --- 初始化相机 (标准流程) ---
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

    print("成功开启视频流！")
    print("=========================================")
    print("【标定指南】")
    print("1. 请将定板平铺在机床桌面上。")
    print("2. 调节焦距和光圈，使画面尽可能清晰。")
    print("3. 按下键盘上的 'c' 键，程序将自动计算并打印比例尺。")
    print("4. 按下 'q' 键退出程序。")
    print("=========================================")

    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    # 亚像素角点优化的停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret == 0:
            nHeight = stOutFrame.stFrameInfo.nHeight
            nWidth = stOutFrame.stFrameInfo.nWidth
            nFrameLen = stOutFrame.stFrameInfo.nFrameLen
            
            data = ctypes.string_at(stOutFrame.pBufAddr, nFrameLen)
            img_array = np.frombuffer(data, dtype=np.uint8)
            cv_image = img_array.reshape((nHeight, nWidth)).copy() # 必须 copy，防止只读报错
            
            # 转为彩色用于显示
            display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
            cv2.resizeWindow("Calibration", 800, 640)
            cv2.imshow("Calibration", display_image)

            cam.MV_CC_FreeImageBuffer(stOutFrame)

            # 监听键盘指令
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('c'):
                print("\n开始提取角点，请稍候...")
                # 寻找棋盘格角点
                ret_board, corners = cv2.findChessboardCorners(cv_image, CHECKERBOARD, None)
                
                if ret_board:
                    # 如果找到了，使用亚像素级精度进一步优化角点坐标
                    corners2 = cv2.cornerSubPix(cv_image, corners, (11, 11), (-1, -1), criteria)
                    
                    # 在画面上画出彩色角点连线，直观验证是否找对
                    cv2.drawChessboardCorners(display_image, CHECKERBOARD, corners2, ret_board)
                    cv2.imshow("Calibration", display_image)
                    cv2.waitKey(500) # 停顿0.5秒让你看清连线

                    # --- 核心计算逻辑：计算相邻角点之间的平均像素距离 ---
                    pixel_distances = []
                    # 计算横向相邻角点的距离
                    for i in range(CHECKERBOARD[1]):
                        for j in range(CHECKERBOARD[0] - 1):
                            idx1 = i * CHECKERBOARD[0] + j
                            idx2 = idx1 + 1
                            pt1 = corners2[idx1][0]
                            pt2 = corners2[idx2][0]
                            dist = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                            pixel_distances.append(dist)
                            
                    avg_pixel_dist = sum(pixel_distances) / len(pixel_distances)
                    
                    # 得出最终比例尺：物理尺寸 / 像素尺寸
                    Ratio_table = SQUARE_SIZE_MM / avg_pixel_dist
                    
                    print(f"【标定成功】")
                    print(f"-> 画面中单格平均像素边长: {avg_pixel_dist:.2f} pixels")
                    print(f"-> 绝对物理比例尺 (Ratio_table): {Ratio_table:.6f} mm/pixel")
                    print("请将此数值复制并填入你的主检测代码中！\n")
                else:
                    print("【标定失败】未在画面中找到完整的棋盘格！请检查：")
                    print("1. 标定板是否部分超出了相机视野？")
                    print("2. 标定板是否过曝反光？")
                    print("3. CHECKERBOARD (内角点数量) 设置是否与实际可见的格子对应？")

    # 收尾
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()