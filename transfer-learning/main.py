from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")

img_path = "10.jpg"

img = Image.open(img_path)

results = model.predict(img, save=True)

for result in results:
    boxes = result.boxes
    print(f"Resimde toplam {len(boxes)} adet nesne tespit edildi.")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf_score = box.conf[0]
        print(conf_score)

# 