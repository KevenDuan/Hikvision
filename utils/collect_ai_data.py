# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import ctypes
import pandas as pd
import time

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    sys.exit()

def main():
    # =========================================================
    # 【采集配置】
    # =========================================================
    # 1. 你当前用来采集的目标物的真实物理数据
    # 如果此时你放的是立方体，高度填 51.0
    # 如果此时你放的是桌面纸片，高度填 0.0
    ACTUAL_OBJECT_HEIGHT = 0
    CAD_LENGTH_MM = 106  # 物品对应的物理长边
    CAD_WIDTH_MM = 89.0   # 物品对应的物理短边
    
    # 2. 图像画面尺寸 (用于特征归一化，防止图像分辨率影响模型)
    IMG_W, IMG_H = 1280, 960
    # =========================================================

    # 初始化相机
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
    cam.MV_CC_StartGrabbing()

    print("\n" + "="*50)
    print(f"【AI 数据采集系统】当前高度标签: {ACTUAL_OBJECT_HEIGHT}mm")
    print("操作引导：")
    print("1. 移动物品到画面不同位置（中心、边缘、角落）。")
    print("2. 按 's' 键保存一组特征数据。")
    print("3. 收集至少 100 组以上数据后，按 'q' 保存并退出。")
    print("="*50 + "\n")

    data_list = []
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

            # --- OpenCV 特征提取逻辑 (确保与实时检测代码一致！) ---
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_features = None
            for cnt in contours:
                if cv2.contourArea(cnt) > 2000: # 面积过滤
                    rect = cv2.minAreaRect(cnt)
                    box = np.int32(cv2.boxPoints(rect))
                    cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
                    
                    (cx, cy), (w_pixel, h_pixel), angle = rect
                    long_px = max(w_pixel, h_pixel)
                    short_px = min(w_pixel, h_pixel)

                    # 构造 AI 需要的特征向量 (输入 X)
                    # 我们希望 AI 学习到：不同位置(cx,cy)下的像素是如何“虚胖”或“被畸变扯大”的
                    current_features = {
                        'norm_cx': cx / IMG_W,      # 归一化中心 X
                        'norm_cy': cy / IMG_H,      # 归一化中心 Y
                        'long_px': long_px,         # 像素长边
                        'short_px': short_px,       # 像素短边
                        'cad_ratio': CAD_LENGTH_MM / CAD_WIDTH_MM, # 物品本身的长宽比
                        'area_px': cv2.contourArea(cnt)
                    }
                    break # 只采集第一个符合的目标

            # 屏幕显示已收集数量
            cv2.putText(display_image, f"Collected: {len(data_list)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.namedWindow("AI Data Collection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("AI Data Collection", 800, 640)
            cv2.imshow("AI Data Collection", display_image)
            cam.MV_CC_FreeImageBuffer(stOutFrame)

            # 按键监听
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_features is not None:
                # 给特征加上“真实高度”标签 (输出 y)
                current_features['true_height_mm'] = ACTUAL_OBJECT_HEIGHT
                data_list.append(current_features)
                print(f"[Save] Data Point {len(data_list)}: At ({current_features['norm_cx']:.2f}, {current_features['norm_cy']:.2f}) -> Height: {ACTUAL_OBJECT_HEIGHT}")
                # 闪烁效果提示已保存
                display_image[:] = 255
                cv2.imshow("AI Data Collection", display_image)
                cv2.waitKey(50)

    # 保存数据为 CSV 文件
    if data_list:
        df = pd.DataFrame(data_list)
        # 尝试追加到现有文件，如果没有则新建
        try:
            existing_df = pd.read_csv('visual_height_dataset.csv')
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv('visual_height_dataset.csv', index=False)
        print(f"\n【采集完成】成功将 {len(data_list)} 组新数据追加保存至 visual_height_dataset.csv，当前总数据量: {len(df)}。")
    
    # 收尾
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()