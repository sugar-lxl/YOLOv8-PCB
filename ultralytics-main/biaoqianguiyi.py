import os
from PIL import Image
 
def get_image_size(image_file):
    # 使用PIL库获取图片尺寸信息
    with Image.open(image_file) as img:
        width, height = img.size
    return width, height
 
def process_label_files(image_dir, label_dir):
    # 遍历图片目录
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_filename = os.path.join(image_dir, filename)
            label_filename = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
 
            # 如果对应的标签文件存在，则进行处理
            if os.path.exists(label_filename):
                with open(label_filename, 'r') as f:
                    lines = f.readlines()
 
                # 获取当前图片的尺寸信息
                image_width, image_height = get_image_size(image_filename)
 
                # 针对每行进行处理
                normalized_lines = []
                for line in lines:
                    line = line.strip()
                    parts = line.split()
                    label = int(parts[0])  # 提取类别标签
                    coordinates = [float(x) for x in parts[1:]]  # 提取坐标信息，并转换为浮点数
                    num_boxes = len(coordinates) // 4  # 计算矩形框的数量
                    normalized_coordinates = []
 
                    # 对每个矩形框的坐标进行归一化处理，并转换为 YOLOv5 标签文件格式
                    for i in range(num_boxes):
                        x1 = coordinates[i * 4]
                        y1 = coordinates[i * 4 + 1]
                        x2 = coordinates[i * 4 + 2]
                        y2 = coordinates[i * 4 + 3]
                        x_center = (x1 + x2) / 2 / image_width
                        y_center = (y1 + y2) / 2 / image_height
                        box_width = (x2 - x1) / image_width
                        box_height = (y2 - y1) / image_height
                        normalized_coordinates.append((x_center, y_center, box_width, box_height))
 
                    # 将处理后的结果添加到列表中
                    normalized_line = ' '.join([str(label)] + [' '.join(map(lambda x: "{:.6f}".format(x), coord)) for coord in normalized_coordinates])
                    normalized_lines.append(normalized_line)
 
                # 将处理后的结果写回文件（使用 YOLOv5 标签文件格式）
                with open(label_filename, 'w') as f:
                    f.write('\n'.join(normalized_lines))
 
# 图片目录路径
image_dir = r'C:/Users/lxl/Desktop/finish-yolo/ultralytics-main/deeppcb1/valid/images'
# 标签目录路径
label_dir = r'C:/Users/lxl/Desktop/finish-yolo/ultralytics-main/deeppcb1/valid/labels'
 
# 处理图片和标签文件
process_label_files(image_dir, label_dir)