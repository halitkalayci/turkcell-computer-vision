from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt") #Fine-Tune edilmiş versiyon

IMG_SIZE = 640

img = Image.open("1.jpg")
img = img.resize((IMG_SIZE, IMG_SIZE))

results = model.predict(img, save=True)

for result in results:
    boxes = result.boxes
    print(f"Resimde toplam {len(boxes)} adet nesne tespit edildi.")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf_score = box.conf[0]
        print(conf_score)
