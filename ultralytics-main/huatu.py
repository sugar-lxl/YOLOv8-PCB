import matplotlib.pyplot as plt

# 数据
data = {
    "YOLOv5n": {"mAP": 95.0, "size": 5.3, "color": "blue", "marker": "o"},
    "YOLOv8n": {"mAP": 95.0, "size": 6.5, "color": "green", "marker": "s"},
    "RT-DETR": {"mAP": 98.6, "size": 40.0, "color": "red", "marker": "D"},
    "YOLOv10-n": {"mAP": 96.1, "size": 5.8, "color": "purple", "marker": "P"},
    "Ours": {"mAP": 96.3, "size": 2.5, "color": "orange", "marker": "*"},
}

# 绘制散点图
plt.figure(figsize=(10, 7))
for model, attributes in data.items():
    plt.scatter(
        attributes["size"], attributes["mAP"],
        label=model, color=attributes["color"],
        marker=attributes["marker"], s=120  # 点的大小
    )

# 图表设置
# plt.title("Model Comparison: mAP vs. Model Size", fontsize=24)
plt.xlabel("Model Size (MB)", fontsize=24)
plt.ylabel("mAP@0.5/%", fontsize=24)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Models", fontsize=18, title_fontsize=18,loc='lower right')
plt.xticks(fontsize=18)  # 横坐标刻度字体大小
plt.yticks(fontsize=18)  # 纵坐标刻度字体大小
plt.grid(True, linestyle='None')  # 去除网格线
plt.show()  # 显示图表