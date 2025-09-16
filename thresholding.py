# Eşikleme -> Bir eşik belirle altında-üstünde değerleri farklı renk olarak ifade et.
import cv2
from numpy.char import lower

img = cv2.imread("kedi.jpg")
img = cv2.resize(img, None, fx=0.5,fy=0.5)
# Blur ekleme sebebimiz gürültüyü azaltmak.
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_img = cv2.GaussianBlur(grayscale_img, (9,9), 0)


_, thresholded_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)

import numpy as np
median_val = np.median(blurred_img)
sigma = 0.33
lower_thresh = int(max(0, (1 - sigma) * median_val))
upper_thresh = int(min(255, (1 + sigma) * median_val))


edges = cv2.Canny(blurred_img, lower_thresh, upper_thresh)
print(edges)

cv2.imshow("Edges", edges)
cv2.imshow("Thresholded Kedi", thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Connected Components