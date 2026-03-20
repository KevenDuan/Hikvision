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
    # 【核心配置区】请填入你之前实验测出来的真实数据！
    # =========================================================
    # 1. 相机环境绝对参数 (从前两个脚本测得)
    H_CAM = 363.68         # 刚才用双平面法算出来的相机绝对光学高度 (单位: mm)
    RATIO_TABLE = 0.225366    # 之前在桌面上标定出来的比例尺 (单位: mm/pixel)
    
    # 2. 你的测试物品 CAD 尺寸 (用尺子量一下你手边的测试物)
    CAD_LENGTH_MM = 31.0 # 物品的真实长边 (例如：手机长度 160mm)
    CAD_WIDTH_MM = 31.0   # 物品的真实短边 (例如：手机宽度 75mm)
    
    # --- 数学引擎：计算相机的焦距常数 K ---
    # K 的物理意义：在这台相机里，1毫米的物体在 1毫米的距离上，看起来是多少个像素。
    K_FOCAL = H_CAM / RATIO_TABLE 
    # =========================================================

    # 初始化相机 (标准流程)
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

    print("\n" + "="*50)
    print("【CAD 先验尺寸 3D 高度反推系统】")
    print(f"载入目标尺寸: {CAD_LENGTH_MM}mm x {CAD_WIDTH_MM}mm")
    print("正在实时侦测并反推高度...")
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

            # --- OpenCV 图像处理：提取轮廓 ---
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            # 注意：如果你的测试物是深色的放在浅色桌面上，请改成 cv2.THRESH_BINARY_INV
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000: # 过滤杂质，由于目标通常较大，这里的面积阈值设高一点
                    # 获取最小外接矩形
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w_pixel, h_pixel), angle = rect
                    
                    # 区分长短边 (防止物品旋转导致长宽颠倒)
                    measured_long_px = max(w_pixel, h_pixel)
                    measured_short_px = min(w_pixel, h_pixel)
                    
                    if measured_short_px > 0:
                        # --- 核心黑科技：利用 CAD 宽度和像素宽度，反推镜头到物品顶部的距离 (Z) ---
                        # 公式: Z = (K * 真实宽度) / 像素宽度
                        distance_to_top = (K_FOCAL * CAD_WIDTH_MM) / measured_short_px
                        
                        # --- 得出最终高度 ---
                        # 物品高度 = 相机到桌面的总高度 - 相机到物品顶部的高度
                        object_height = H_CAM - distance_to_top

                        # 画框与显示数据
                        box = np.int32(cv2.boxPoints(rect))
                        cv2.drawContours(display_image, [box], 0, (255, 0, 0), 2)
                        
                        # 在中心打印推算出的 Z 轴高度 (使用醒目的黄色)
                        cv2.putText(display_image, f"Height (Z): {object_height:.1f} mm", 
                                    (int(cx) - 80, int(cy) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # 打印距离镜头的距离作为参考
                        cv2.putText(display_image, f"Dist to Lens: {distance_to_top:.1f} mm", 
                                    (int(cx) - 80, int(cy) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.namedWindow("3D Height Estimation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("3D Height Estimation", 800, 640)
            cv2.imshow("3D Height Estimation", display_image)
            
            cam.MV_CC_FreeImageBuffer(stOutFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 收尾
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()