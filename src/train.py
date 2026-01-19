from ultralytics import YOLO
import torch

device_id = 0
device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device_str}')
batch_size = 32 

model = YOLO('./yolo11n.pt')

results = model.train(
    data='../data/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=batch_size,
    device=device_str,
    plots=False,
    workers=4 
)

print("Training and validation completed.")
