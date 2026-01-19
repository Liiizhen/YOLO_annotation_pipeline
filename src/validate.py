from ultralytics import YOLO
import torch

model = YOLO('runs/detect/train/weights/best.pt')

device_id = 0
device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

metrics = model.val(
    data='../data/data.yaml', 
    device=device_str
)

print(f"mAP (IoU=0.5:0.95): {metrics.box.map}")
print(f"mAP50 (IoU=0.5):    {metrics.box.map50}")
