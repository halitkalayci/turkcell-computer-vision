import cv2
import numpy as np

def detect_plate(img_path):
    """
    Yapay zeka kullanmadan, yalnızca OpenCV ile bir görüntüdeki plakayı tespit eden fonksiyon.

    Args:
        img_path (str): İşlenecek resmin yolu.
    
    Returns:
        None: Sonuçlar aynı pencerelerde gösterilecek.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Resim yüklenemedi -> {img_path}")
        return
    
    ratio = 620 / img.shape[1] #genişlik 620 olduğunda yükseklik ne kadar küçülmeli (oran)
    dim = (620, int(img.shape[0] * ratio)) # hesapla.
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    grayscale_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Bilateral Filter -> Gürültüyü azaltmak için kullanılır.
    # d -> komşuluk çapı
    # sigmaColor -> renk farkı duyarlılığı
    # sigmaSpace -> uzamsal farkı duyarlılığı
    filtered_img = cv2.bilateralFilter(grayscale_img, 11, 17, 17)
    cv2.imshow("Filtered Image", filtered_img)

    edged = cv2.Canny(filtered_img, 50, 200)
    cv2.imshow("Edged Image", edged)

    contours, hiearachy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Konturları alanı büyükten küçüğe sırala.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    all_contours_image = img_resized.copy()
    # -1 -> Hiyerarşideki tüm konturları çiz.
    cv2.drawContours(all_contours_image, contours, -1, (0,0,255), 1)
    cv2.imshow("All Contours Image", all_contours_image)

    plate_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Douglas-Peucker
        if len(approx) == 4:
            print("4 köşeli plaka adayı bulundu.")
            (x,y,w,h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)

            if aspect_ratio > 2.5 and aspect_ratio < 5.5 and area > 1000: # araba? arka cam?
                plate_contour = approx
                break

    output_image = img_resized.copy()
    if plate_contour is not None:
        cv2.drawContours(output_image, [plate_contour], -1, (0, 255, 0), 3)
        print(f"'{img_path}' için potansiyel plaka bulundu.")
    else:
        print(f"'{img_path}' için plaka tespit edilemedi.")
    
    cv2.imshow(f"Adim 5b: SONUC - Tespit Edilen Plaka - {img_path}", output_image)
    
    # Crop only plate
    if plate_contour is not None:
        (x,y,w,h) = cv2.boundingRect(plate_contour)
        plate_img = img_resized[y:y+h, x:x+w]
        cv2.imshow("Plate Image", plate_img)
    else:
        print(f"'{img_path}' için plaka tespit edilemedi.")
    


    cv2.waitKey(0)
    cv2.destroyAllWindows()

#detect_plate("data/1.jpg");
detect_plate("data/1002.jpg");

#Yapay Zeka -> Entegre
# Neural Network -> Sinir Ağları
# CNN -> Convolutional Neural Network

# 1- transfer-learning - fine-tuning
# 2- tamamen sıfırdan kendi modelimizi üretmek
