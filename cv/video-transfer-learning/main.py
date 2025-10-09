# Video -> Frame00000.png -> ndarray
# Video -> Frame00001.png -> ndarray
# Video -> Frame00002.png -> ndarray
# Video -> Frame00003.png -> ndarray

import cv2
import os

VIDEO_PATH = "video.mp4"
OUT_DIR = "data/images/train"
FRAME_STRIDE = 5 # her 5. kareyi al.
LIMIT = None #None limitsiz 200 -> 200 fotoğrafta dur.

os.makedirs(OUT_DIR, exist_ok=True)

capture = cv2.VideoCapture(VIDEO_PATH)
assert capture.isOpened(), f"Video açılamadı {VIDEO_PATH}"

count, saved = 0, 0
while True:
    ok, frame = capture.read()
    if not ok:
        break
    if count % FRAME_STRIDE == 0:
        out_name = f"frame_{saved:06d}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if LIMIT and saved >= LIMIT:
            break
    count += 1

capture.release()
print(f"Frame kayıtları bitti kaydedilen dosya sayısı {saved} okunan frame sayısı: {count}")