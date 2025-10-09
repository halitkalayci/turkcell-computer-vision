from ultralytics import YOLO


model = YOLO("yolov8n.pt")

result = model.predict(
    source="data/images/train",
    imgsz=640,
    conf=0.1,
    save=True,
    save_txt=True,
    save_conf=True,
    verbose=True
)