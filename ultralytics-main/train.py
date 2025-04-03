import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/lxl/Desktop/yolov8/ultralytics-main/ultralytics/cfg/models/v8-New/C2f_Star_Neck-GS_Detect.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='C:/Users/lxl/Desktop/yolov8/PCB_Data/pcb_data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=30,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD13773377861
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/Experiment',
                name='yolov8-MPDIOU-C2f_Star-GS_Head',  
                )