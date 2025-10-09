from collections import deque
import cv2
import time
import numpy as np

def open_source():
    """
    OpenCV'nin örnek videosunu (vtest.avi) açmayı dene.
    """
    try:
        sample_path = cv2.samples.findFile(r"C:\Users\PC1\Desktop\Projects\Personal\Python\cv\video\vtest.avi")
        capture = cv2.VideoCapture(sample_path)
        if capture.isOpened():
            print("[INFO] Kaynak: OpenCV samples/vtest.avi")
            return capture
    except Exception as e:
        print(f"[ERROR] Video açılamadı: {e}")
        return None
# Aspect => Oran
def resize_keep_aspect(frame, max_w=960):
    """
    Performans ve ekranda rahat izleme için genişliği max 960px'e ölçekleyelim.
    """
    h,w = frame.shape[:2]

    if w <= max_w:
        return frame

    scale = max_w / float(w)    
    new_size = (int(w*scale), int(h*scale))
    return cv2.resize(frame, new_size)

MODE_RAW = 1
MODE_GRAY = 2
MODE_MOTION = 3
MODE_BACKGROUND = 4

bg = cv2.createBackgroundSubtractorMOG2(
    history=300, #model geçmişi => daha yüksek = daha yumuşak arkaplan kaldırma
    varThreshold=25, # piksel sınıflandırma eşiği
    detectShadows=True # gölge algılama
)


def main():
    capture = open_source()
    title = "VIDEO ISLEME GIRIS"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1000, 500)    

    fps = capture.get(cv2.CAP_PROP_FPS)
    print("[INFO] FPS: ", fps)
    delay = int(1000/fps) # 100ms

    # FPS ölçümü için değerler.
    fps_hist = deque(maxlen=20)
    t_prev = time.time()

    prev_gray = None
    while True:
        ok, frame = capture.read()
        #capture.grab() # bir kare atla.
        if not ok:
            print("[INFO] Kaynak bitti veya kamera akışı kesildi.")
            break
        frame = resize_keep_aspect(frame)

        now = time.time()
        fps_hist.append(1.0 / (now - t_prev) if now > t_prev else 0.0)
        t_prev = now
        fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

        display = frame.copy()

        mode = MODE_BACKGROUND

        if mode == MODE_GRAY:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        elif mode == MODE_MOTION:
            #Basit hareket algılama algoritması:
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diff = cv2.GaussianBlur(diff, (5, 5), 0)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 400:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 140, 255), 2)
            # Bir sonraki frame için mevcut gray'i sakla
            prev_gray = gray.copy()
        elif mode == MODE_BACKGROUND:
            mask = bg.apply(display)
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display = np.hstack([display, mask_vis])

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imshow(title, display)

        key = cv2.waitKey(delay) & 0xFF # 1ms arayla kullanıcı tuşa basmıyosa 1ms aralıkta bekle ve devam et.

        if key == ord("q") or key == 27:
            print("[INFO] Çıkış yapılıyor...")
            break

    capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()