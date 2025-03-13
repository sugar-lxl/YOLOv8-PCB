# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt

# # 当前路径
# pwd = os.getcwd()

# # 方法名称和颜色设置
# names = ['CIoU', 'DIoU',  'EIoU', 'SIoU', 'Inner-MPDIoU']
# colors = ['blue', 'red', 'purple', 'green', 'orange']  # 不同方法对应的颜色

# # 创建图形
# plt.figure(figsize=(8, 6))  # 设置图的大小

# # 读取并绘制数据
# for i, color in zip(names, colors):  # 使用不同颜色
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
    
#     # 处理 mAP@0.5 数据，去除无效值
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    
#     # 对数据进行采样（每隔5个点取一个），减少密度
#     sampled_epochs = data.index[::5]
#     sampled_mAP = data['       metrics/mAP50(B)'][::5]
    
#     # 绘制曲线
#     plt.plot(sampled_epochs, sampled_mAP, label=i, color=color, linewidth=1.0)

# # 图表优化
# plt.xlabel('Epoch', fontsize=24)  # 设置X轴标签和字体大小
# plt.ylabel('mAP@0.5', fontsize=24)  # 设置Y轴标签和字体大小

# # 调整刻度数字的大小
# plt.tick_params(axis='both', which='major', labelsize=18)  # 设置主刻度数字大小
# plt.tick_params(axis='both', which='minor', labelsize=18)  # 设置次刻度数字大小（如果有）

# # 调整坐标轴的范围
# plt.xlim(50, 300)  # 设置横坐标的范围，0到300
# plt.ylim(0.7, 1)    # 设置纵坐标的范围，0到1

# # plt.title('Comparison of mAP@0.5 Across Different Loss Functions', fontsize=14)  # 设置图标题
# plt.legend(loc='lower right', fontsize=18)  # 图例位置和字体大小
# plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格
# plt.tight_layout()  # 自动调整布局避免重叠

# # 保存图像
# output_path = os.path.join(pwd, 'mAP50_curve_colored.png')
# plt.savefig(output_path, dpi=300)  # 设置高分辨率
# print(f'mAP50_curve_colored.png saved in {output_path}')

# # 显示图形
# plt.show()


#平滑曲线
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import savgol_filter

# 当前路径
pwd = os.getcwd()

# 方法名称和颜色设置
names = ['CIoU', 'DIoU',  'EIoU', 'SIoU', 'Inner-MPDIoU']
colors = ['blue', 'red', 'purple', 'green', 'orange']  # 不同方法对应的颜色

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
    plt.plot(sampled_epochs, smoothed_mAP, label=i, color=color, linewidth=1.0)

# 图表优化
plt.xlabel('Epoch', fontsize=20)  # 设置X轴标签和字体大小
plt.ylabel('mAP@0.5', fontsize=20)  # 设置Y轴标签和字体大小

# 调整刻度数字的大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置主刻度数字大小
plt.tick_params(axis='both', which='minor', labelsize=14)  # 设置次刻度数字大小（如果有）

# 调整坐标轴的范围
plt.xlim(50, 300)  # 设置横坐标的范围，0到300
plt.ylim(0.75, 1)    # 设置纵坐标的范围，0到1

# plt.title('Comparison of mAP@0.5 Across Different Loss Functions', fontsize=14)  # 设置图标题
plt.legend(loc='lower right', fontsize=18)  # 图例位置和字体大小

plt.tight_layout()  # 自动调整布局避免重叠

# 保存图像
output_path = os.path.join(pwd, 'mAP50_curve_colored.png')
plt.savefig(output_path, dpi=300)  # 设置高分辨率
print(f'mAP50_curve_colored.png saved in {output_path}')

# 显示图形
plt.show()


# 优化区分版
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# from scipy.signal import savgol_filter

# # 当前路径
# pwd = os.getcwd()

# # 方法名称、颜色设置以及标记样式
# names = ['CIoU', 'DIoU', 'EIoU', 'SIoU', 'Inner-MPDIoU']
# colors = ['blue', 'red', 'purple', 'green', 'black']  # 选用更易区分的颜色，黑色突出Inner-MPDIoU
# markers = ['o', '^', 's', 'D', 'x']  # 每条曲线不同的标记样式

# # 创建图形
# plt.figure(figsize=(8, 6))  # 设置图的大小

# # 读取并绘制数据
# for i, (color, marker) in zip(names, zip(colors, markers)):
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
    
#     # 处理 mAP@0.5 数据，去除无效值
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    
#     # 提取完整的 x 轴和 y 轴数据
#     epochs = data.index
#     mAP_values = data['       metrics/mAP50(B)']
    
#     # 使用 Savitzky-Golay 滤波器平滑 mAP 曲线
#     if len(mAP_values) >= 11:
#         smoothed_mAP = savgol_filter(mAP_values, window_length=11, polyorder=3)
#     else:
#         smoothed_mAP = mAP_values
    
#     # 绘制平滑后的曲线
#     plt.plot(epochs, smoothed_mAP, label=i, color=color, linewidth=1.5)
    
#     # 仅保留5个标记点
#     num_markers = 5
#     marker_indices = np.linspace(0, len(epochs) - 1, num_markers, dtype=int)
#     plt.scatter(epochs[marker_indices], smoothed_mAP[marker_indices], color=color, marker=marker, s=50, label="_nolegend_")

# # 图表优化
# plt.xlabel('Epoch', fontsize=20)
# plt.ylabel('mAP@0.5', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.xlim(50, 300)
# plt.ylim(0.75, 1)
# plt.legend(loc='lower right', fontsize=18)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()

# # 保存图像
# output_path = os.path.join(pwd, 'mAP50_curve_updated.png')
# plt.savefig(output_path, dpi=300)
# print(f'mAP50_curve_updated.png saved in {output_path}')

# # 显示图形
# plt.show()
# #

##
# # 平滑曲线
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# from scipy.signal import savgol_filter

# # 当前路径
# pwd = os.getcwd()

# # 方法名称和颜色设置
# names = ['CIoU', 'DIoU', 'EIoU', 'SIoU', 'Inner-MPDIoU']
# colors = ['cyan', 'magenta', 'yellow', 'lime', 'pink']  # 更鲜艳的颜色

# # 创建图形
# plt.figure(figsize=(10, 6))  # 设置图的大小

# # 读取并绘制数据
# for i, color in zip(names, colors):  # 使用不同颜色
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
    
#     # 处理 mAP@0.5 数据，去除无效值
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
#     data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    
#     # 对数据进行采样（每隔N个点取一个），减少密度
#     N = len(data) // 5  # 只保留5个点
#     sampled_epochs = data.index[::N]
#     sampled_mAP = data['       metrics/mAP50(B)'][::N]
    
#     # 使用 Savitzky-Golay 滤波器平滑 mAP 曲线
#     # 增大窗口长度，降低多项式阶数，确保曲线平滑
#     window_length = min(51, len(sampled_mAP))  # 窗口长度不超过数据点数量
#     if window_length % 2 == 0:  # 窗口长度必须是奇数
#         window_length -= 1
#     smoothed_mAP = savgol_filter(sampled_mAP, window_length=window_length, polyorder=2)  # 降低多项式阶数
    
#     # 绘制平滑后的曲线
#     plt.plot(sampled_epochs, smoothed_mAP, label=i, color=color, linewidth=2.0, marker='o', markersize=8)  # 添加标记

# # 图表优化
# plt.xlabel('Epoch', fontsize=24)  # 设置X轴标签和字体大小
# plt.ylabel('mAP@0.5', fontsize=24)  # 设置Y轴标签和字体大小

# # 调整刻度数字的大小
# plt.tick_params(axis='both', which='major', labelsize=18)  # 设置主刻度数字大小
# plt.tick_params(axis='both', which='minor', labelsize=18)  # 设置次刻度数字大小（如果有）

# # 调整坐标轴的范围
# plt.xlim(50, 300)  # 设置横坐标的范围，0到300
# plt.ylim(0.7, 1)    # 设置纵坐标的范围，0到1

# # plt.title('Comparison of mAP@0.5 Across Different Loss Functions', fontsize=14)  # 设置图标题
# plt.legend(loc='lower right', fontsize=18)  # 图例位置和字体大小

# plt.tight_layout()  # 自动调整布局避免重叠

# # 保存图像
# output_path = os.path.join(pwd, 'mAP50_curve_colored.png')
# plt.savefig(output_path, dpi=300)  # 设置高分辨率
# print(f'mAP50_curve_colored.png saved in {output_path}')

# # 显示图形
# plt.show()