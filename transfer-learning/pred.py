from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt") #Fine-Tune edilmiş versiyon


img = Image.open("2.jpg")

results = model.predict(source=img, save=True, imgsz=640, conf=0.25)

for result in results:
    boxes = result.boxes
    print(f"Resimde toplam {len(boxes)} adet nesne tespit edildi.")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf_score = box.conf[0]
        print(conf_score)
