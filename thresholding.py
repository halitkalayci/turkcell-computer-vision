# Eşikleme -> Bir eşik belirle altında-üstünde değerleri farklı renk olarak ifade et.
import cv2
from numpy.char import lower

img = cv2.imread("kedi.jpg")
img = cv2.resize(img, None, fx=0.5,fy=0.5)
# Blur ekleme sebebimiz gürültüyü azaltmak.
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_img = cv2.GaussianBlur(grayscale_img, (9,9), 0)


_, thresholded_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)

# src -> image
# connectivity -> 4,8
# ltype -> cv2.CV_32S -> tam sayı etiket türü , 0,1,2,3
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, 4, cv2.CV_32S)

print("num_labels", num_labels)
print("labels", labels)
print("stats", stats)
print("centroids", centroids)

min_area = 400
obj_count = 0
output_img = img.copy()

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area > min_area:
        obj_count += 1
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        right = left + width
        bottom = top + height
        
        #Verilen alana, 2 kalınlığında (border) bir rectangle çiz.
        cv2.rectangle(output_img, (left, top), (right, bottom), (0, 0, 255), 2)
        #cv2.putText(img, f"Obj {obj_count}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print("obj_count", obj_count)
cv2.imshow("Output Image", output_img)


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
