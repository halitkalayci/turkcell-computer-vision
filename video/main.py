from collections import deque
import cv2
import time

def open_source():
    """
    OpenCV'nin örnek videosunu (vtest.avi) açmayı dene.
    """
    try:
        sample_path = cv2.samples.findFile("vtest.avi")
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
        print("[INFO] FPS: ", fps)

        display = frame.copy()

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 28), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (20, 220, 20), 
        2, cv2.LINE_AA)
        
        cv2.imshow(title, display)
  

        key = cv2.waitKey(delay) & 0xFF # 1ms arayla kullanıcı tuşa basmıyosa 1ms aralıkta bekle ve devam et.

        if key == ord("q") or key == 27:
            print("[INFO] Çıkış yapılıyor...")
            break

    capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()