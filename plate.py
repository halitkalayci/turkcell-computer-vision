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



    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_plate("data/1.jpg");