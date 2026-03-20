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
    print("【错误】找不到 MvImport 库！请确保 MvImport 文件夹和本脚本在同一目录下。")
    sys.exit()

def get_avg_pixel_distance(gray_img, checkerboard_size):
    """
    核心测算函数：寻找棋盘格并计算相邻角点的平均像素距离
    """
    # 寻找内角点
    ret, corners = cv2.findChessboardCorners(gray_img, checkerboard_size, None)
    if not ret:
        return False, 0.0, None

    # 亚像素级精度优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)

    # 计算横向相邻角点之间的平均像素距离
    distances = []
    cols, rows = checkerboard_size
    for i in range(rows):
        for j in range(cols - 1):
            idx1 = i * cols + j
            idx2 = idx1 + 1
            pt1 = corners_subpix[idx1][0]
            pt2 = corners_subpix[idx2][0]
            dist = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            distances.append(dist)
            
    avg_dist = sum(distances) / len(distances)
    return True, avg_dist, corners_subpix

def main():
    # =========================================================
    # 必须要确认的物理实验参数！
    # =========================================================
    CHECKERBOARD = (9, 6)  # A4 纸 6x9 方块对应的内角点数量
    BLOCK_HEIGHT_MM = 16 # 【关键】你用来垫高标定纸的垫块/书本的精准厚度 (单位: 毫米)
    # =========================================================

    # 初始化相机
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
    cam.MV_CC_SetFloatValue("ExposureTime", 10000)
    cam.MV_CC_StartGrabbing()

    print("\n" + "="*50)
    print("【Z 轴光学高度自动标定系统】")
    print(f"当前设定的垫块高度: {BLOCK_HEIGHT_MM} mm")
    print("操作步骤：")
    print("1. 将 A4 标定纸平放在【桌面】上，按 '1' 键记录底层数据。")
    print("2. 将垫块放在相机下方，把标定纸平放在【垫块】上，按 '2' 键记录高层数据。")
    print("3. 按 'c' 键计算相机的绝对光学高度。")
    print("4. 按 'q' 键退出。")
    print("="*50 + "\n")

    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    # 用于存储两次测量的像素数值
    p_table = 0.0
    p_block = 0.0

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

            # 在屏幕上显示当前的测量状态
            status_text = f"P_table (1): {p_table:.2f} px | P_block (2): {p_block:.2f} px"
            cv2.putText(display_image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.namedWindow("Height Calibration", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Height Calibration", 800, 640)
            cv2.imshow("Height Calibration", display_image)
            cam.MV_CC_FreeImageBuffer(stOutFrame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('1'):
                print("-> 正在计算桌面层的像素大小...")
                success, avg_dist, corners = get_avg_pixel_distance(cv_image, CHECKERBOARD)
                if success:
                    p_table = avg_dist
                    print(f"[成功] 桌面层平均像素边长 (P_table) = {p_table:.2f}")
                else:
                    print("[失败] 无法提取桌面层角点，请检查画面是否清晰、标定纸是否完全在视野内。")

            elif key == ord('2'):
                print("-> 正在计算垫块层的像素大小...")
                success, avg_dist, corners = get_avg_pixel_distance(cv_image, CHECKERBOARD)
                if success:
                    p_block = avg_dist
                    print(f"[成功] 垫块层平均像素边长 (P_block) = {p_block:.2f}")
                else:
                    print("[失败] 无法提取垫块层角点，请检查标定纸是否超出了相机视野。")

            elif key == ord('c'):
                if p_table > 0 and p_block > 0:
                    if p_block <= p_table:
                        print("\n【逻辑错误】垫块上的像素大小必须大于桌面上的像素大小！请重新测量。")
                    else:
                        # 核心公式计算高度
                        # H = (Δh * P_block) / (P_block - P_table)
                        camera_height = (BLOCK_HEIGHT_MM * p_block) / (p_block - p_table)
                        print("\n" + "*"*50)
                        print(f"【计算完成】")
                        print(f"相机镜头到桌面的绝对物理高度 (H_cam) = {camera_height:.2f} mm")
                        print("*"*50 + "\n")
                else:
                    print("\n【警告】请先完成步骤 1 和步骤 2 的数据采集，再按 'c' 计算！")

    # 收尾
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()