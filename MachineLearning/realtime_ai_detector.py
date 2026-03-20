# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import ctypes
import joblib # 载入模型需要 Joblib
import pandas as pd

# 导入海康 SDK 核心库
sys.path.append("./MvImport")
try:
    from MvCameraControl_class import *
except ImportError:
    sys.exit()

def main():
    # =========================================================
    # 【推理配置】
    # =========================================================
    # 1. 图像画面尺寸 (必须与采集代码一致！)
    IMG_W, IMG_H = 1280, 960
    
    # 2. 载入提前训练好的 AI 模型
    MODEL_FILE = r'C:\Users\27732\Desktop\vision\MachineLearning\visual_height_model.pkl'
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"【错误】找不到模型文件 {MODEL_FILE}，请先训练模型！")
        return
    
    # 定义模型需要的特征列名 (顺序必须与训练时一致！)
    FEATURE_COLS = ['norm_cx', 'norm_cy', 'long_px', 'short_px', 'cad_ratio', 'area_px']
    # =========================================================

    # 初始化相机 (标准流程)
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
    print("【AI 深度优化单目 3D 深度系统】")
    print("正在加载 AI 回归模型进行修正...")
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

            # --- OpenCV 图像处理：提取特征 ---
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000: # 过滤杂质
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w_pixel, h_pixel), angle = rect
                    
                    # 区分长短边
                    long_px = max(w_pixel, h_pixel)
                    short_px = min(w_pixel, h_pixel)
                    
                    if short_px > 0:
                        # --- 构造 AI 特征特征行 ---
                        # 假设我们知道现在测的是 51x51 的立方体，长宽比为 1
                        current_cad_ratio = 51.0 / 51.0 
                        
                        features_dict = {
                            'norm_cx': cx / IMG_W,
                            'norm_cy': cy / IMG_H,
                            'long_px': long_px,
                            'short_px': short_px,
                            'cad_ratio': current_cad_ratio,
                            'area_px': area
                        }
                        
                        # 将特征字典转为 Scikit-learn 需要的 DataFrame 格式
                        features_df = pd.DataFrame([features_dict])[FEATURE_COLS]
                        
                        # --- 黑科技：模型推理 ---
                        predicted_height = model.predict(features_df)[0]

                        # 画框与显示修正后的 Z 轴高度
                        box = np.int32(cv2.boxPoints(rect))
                        cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
                        
                        # 在中心打印推算出的 Z 轴高度 (使用醒目的黄色)
                        cv2.putText(display_image, f"AI Height (Z): {predicted_height:.2f} mm", 
                                    (int(cx) - 80, int(cy) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # 打印像素状态用于排错
                        cv2.putText(display_image, f"Loc: ({int(cx)},{int(cy)}) | PxL: {int(long_px)}", 
                                    (int(cx) - 80, int(cy) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.namedWindow("3D AI Height Estimation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("3D AI Height Estimation", 800, 640)
            cv2.imshow("3D AI Height Estimation", display_image)
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 收尾流程
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()