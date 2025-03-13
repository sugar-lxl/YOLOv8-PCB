# 平滑曲线
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import savgol_filter

# 当前路径
pwd = os.getcwd()

# 方法名称和颜色设置（调整颜色，提高辨识度）
names = ['CIoU', 'DIoU', 'EIoU', 'SIoU', 'Inner-MPDIoU']
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#000000']  # 选取更易区分的颜色

# 创建图形
plt.figure(figsize=(10, 6))  # 设置图的大小

# 读取并绘制数据
for i, color in zip(names, colors):  # 使用不同颜色
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    
    # 处理 mAP@0.5 数据，去除无效值
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    
    # 对数据进行采样（每隔5个点取一个），减少密度
    sampled_epochs = data.index[::5]
    sampled_mAP = data['       metrics/mAP50(B)'][::5]
    
    # 使用 Savitzky-Golay 滤波器平滑 mAP 曲线
    smoothed_mAP = savgol_filter(sampled_mAP, window_length=11, polyorder=3)  # 设置窗口大小和多项式阶数
    
    # 绘制平滑后的曲线
    plt.plot(sampled_epochs, smoothed_mAP, label=i, color=color, linewidth=2.0)  # 增加线宽提高可见性

# 图表优化
plt.xlabel('Epoch', fontsize=20)  # 设置X轴标签和字体大小
plt.ylabel('mAP@0.5', fontsize=20)  # 设置Y轴标签和字体大小

# 调整刻度数字的大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置主刻度数字大小
plt.tick_params(axis='both', which='minor', labelsize=14)  # 设置次刻度数字大小（如果有）

# 调整坐标轴的范围
plt.xlim(50, 300)  # 设置横坐标的范围
plt.ylim(0.85, 1)  # 设置纵坐标的范围

plt.legend(loc='lower right', fontsize=18)  # 图例位置和字体大小

plt.tight_layout()  # 自动调整布局避免重叠

# 保存图像
output_path = os.path.join(pwd, 'mAP50_curve_colored_updated.png')
plt.savefig(output_path, dpi=300)  # 设置高分辨率
print(f'mAP50_curve_colored_updated.png saved in {output_path}')

# 显示图形
plt.show()
