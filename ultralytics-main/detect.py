import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/lxl/Desktop/finish-yolo/runs/deeppcb/yolov8m/weights/best.pt') # select your model.pt path
    model.predict(source='C:/Users/lxl/Desktop/finish-yolo/runs/TuPian/DuiBi',
                  imgsz=640,
                  project='C:/Users/lxl/Desktop/finish-yolo/runs/TuPian/DuiBi-JirGuo',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # visualize=True # visualize model features maps
                )