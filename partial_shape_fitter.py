# -*- coding: utf-8 -*-
"""
防遮挡检测矩形轮廓
"""
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

def get_rho_theta(x1, y1, x2, y2):
    """将两点坐标转换为极坐标系下的 rho 和 theta (法线角度)"""
    dx = x2 - x1
    dy = y2 - y1
    theta_line = math.atan2(dy, dx)
    theta_normal = theta_line + math.pi / 2
    rho = x1 * math.cos(theta_normal) + y1 * math.sin(theta_normal)
    
    # 归一化 theta 到 [0, pi) 范围
    if theta_normal < 0:
        theta_normal += math.pi
        rho = -rho
    elif theta_normal >= math.pi:
        theta_normal -= math.pi
        rho = -rho
    return rho, theta_normal

def intersect_lines(rho1, theta1, rho2, theta2):
    """【核心数学】解二元一次方程组，求两条直线的交点"""
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        # 使用线性代数矩阵求解交点
        x0 = np.linalg.solve(A, b)
        return int(np.round(x0[0][0])), int(np.round(x0[1][0]))
    except np.linalg.LinAlgError:
        return None # 两线平行，无交点

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
    cam.MV_CC_StartGrabbing()

    print("\n" + "="*50)
    print("【360度全姿态 抗遮挡几何重建系统】")
    print("请任意旋转零件，并用手遮挡，见证数学的力量！")
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

            # 1. 边缘检测
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 2. 霍夫概率变换提取线段
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=15)

            if lines is not None and len(lines) > 0:
                # 3. 寻找全场最长的一条线，作为零件倾斜的“绝对基准角”
                longest_len = 0
                base_theta = 0
                all_line_params = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.hypot(x2 - x1, y2 - y1)
                    rho, theta = get_rho_theta(x1, y1, x2, y2)
                    all_line_params.append((rho, theta, length))
                    
                    if length > longest_len:
                        longest_len = length
                        base_theta = theta

                group_A = [] # 平行于基准角的线
                group_B = [] # 垂直于基准角的线

                # 4. 动态分组：不管你怎么转，边只会有横竖两组
                for rho, theta, length in all_line_params:
                    # 计算角度差 (弧度)
                    diff = abs(theta - base_theta)
                    diff = min(diff, math.pi - diff) # 解决 pi 和 0 是一样的情况
                    
                    angle_diff_deg = math.degrees(diff)
                    
                    # 容差设为 15 度，在这个范围内的分入同一组
                    if angle_diff_deg < 15:
                        group_A.append(rho)
                    elif abs(angle_diff_deg - 90) < 15:
                        group_B.append(rho)

                # 5. 寻找最外围的四条物理边界
                if len(group_A) >= 2 and len(group_B) >= 2:
                    rho_A_min, rho_A_max = min(group_A), max(group_A)
                    rho_B_min, rho_B_max = min(group_B), max(group_B)

                    # 为了计算交点，取分组的平均角度作为法线角，过滤掉毛刺带来的微小误差
                    theta_A = base_theta
                    theta_B = base_theta + math.pi/2 if base_theta < math.pi/2 else base_theta - math.pi/2

                    # 6. 【高光时刻】方程矩阵对撞，求出四个顶点的绝对坐标！
                    pt1 = intersect_lines(rho_A_min, theta_A, rho_B_min, theta_B)
                    pt2 = intersect_lines(rho_A_max, theta_A, rho_B_min, theta_B)
                    pt3 = intersect_lines(rho_A_max, theta_A, rho_B_max, theta_B)
                    pt4 = intersect_lines(rho_A_min, theta_A, rho_B_max, theta_B)

                    if pt1 and pt2 and pt3 and pt4:
                        # 确保绘制顺序是一个闭合多边形 (顺时针或逆时针)
                        pts = np.array([pt1, pt2, pt3, pt4], np.int32)
                        
                        # OpenCV 高级绘图：画任意角度的多边形
                        cv2.polylines(display_image, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

                        # 计算绝对中心点
                        center_x = int(sum([p[0] for p in pts]) / 4)
                        center_y = int(sum([p[1] for p in pts]) / 4)

                        # 计算重建后的物理长宽像素
                        w_px = int(math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1]))
                        h_px = int(math.hypot(pt2[0]-pt3[0], pt2[1]-pt3[1]))

                        # 画准星和输出数据
                        cv2.drawMarker(display_image, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(display_image, f"Center: ({center_x}, {center_y})", (center_x + 20, center_y - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(display_image, f"Angle: {int(math.degrees(base_theta))} deg", (center_x + 20, center_y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(display_image, f"PxL: {max(w_px, h_px)}x{min(w_px, h_px)}", (center_x + 20, center_y + 35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 画中画边缘调试
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            small_edges = cv2.resize(edges_bgr, (320, 240))
            display_image[0:240, display_image.shape[1]-320:display_image.shape[1]] = small_edges

            cv2.namedWindow("Any-Angle Geometric Fitter", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Any-Angle Geometric Fitter", 800, 640)
            cv2.imshow("Any-Angle Geometric Fitter", display_image)
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