import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/lxl/Desktop/yolov8/runs/Experiment/YOLOv8n/weights/best.pt')
    model.val(data='C:/Users/lxl/Desktop/yolov8/PCB_Data/pcb_data.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )