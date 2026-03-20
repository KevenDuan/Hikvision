# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    print("\n" + "="*50)
    print("【AI 回归模型训练系统】")
    print("="*50 + "\n")

    # 1. 载入数据集
    try:
        df = pd.read_csv(r'C:\Users\27732\Desktop\vision\MachineLearning\visual_height_dataset.csv')
    except FileNotFoundError:
        print("【错误】找不到 visual_height_dataset.csv 文件，请先运行采集脚本！")
        return

    print(f"-> 成功载入数据集，数据总量: {len(df)}")
    print(df.head())

    # 2. 定义特征（X）和标签（y）
    # 输入特征：norm_cx, norm_cy, long_px, short_px, cad_ratio, area_px
    feature_cols = ['norm_cx', 'norm_cy', 'long_px', 'short_px', 'cad_ratio', 'area_px']
    X = df[feature_cols]
    
    # 预测目标：true_height_mm
    y = df['true_height_mm']

    # 3. 划分训练集和测试集 (80%用于训练，20%用于验证模型准不准)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"-> 正在训练模型... (训练集: {len(X_train)} / 测试集: {len(X_test)})")

    # 4. 选择模型：随机森林回归器
    # 随机森林非常适合这种中等规模数据集，能很好地拟合透视非线性误差，且不易过拟合。
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # 5. 模型训练
    model.fit(X_train, y_train)

    # 6. 模型评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "*"*50)
    print("【训练完成】")
    print(f"-> 平均绝对误差 (MAE): {mae:.2f} mm") # MAE越低越好，比如小于 2.0mm 就蛮准了
    print(f"-> 模型拟合优度 (R2): {r2:.4f}")   # R2越接近 1 越好
    print("*"*50 + "\n")

    # 7. 保存训练好的模型文件
    model_filename = 'visual_height_model.pkl'
    joblib.dump(model, model_filename)
    print(f"-> 模型已被完美固化保存至: {model_filename}")

if __name__ == "__main__":
    main()