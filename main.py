# wrapper
import cv2
# RGB -> Red,Green,Blue
# BGR -> Blue,Green,Red -> OPENCV'nin default okuma stratejisi.
img = cv2.imread("cat.jpg") #BGR

print(type(img))
print(img.shape)
print(img.ndim) 

#bgr->rgb dönüşüm.
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB
print(img2[600,600])

grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Manipülasyon

print(grayscale_img.shape)
print(grayscale_img.ndim)
print(grayscale_img[600,600])


#cv2.imshow("Kedi BGR", img) #BGR
#cv2.imshow("Kedi RGB", img2) #RGB
#cv2.imshow("Kedi Grayscale", grayscale_img) #Grayscale


# Resizing
resized_img = cv2.resize(img, (700,700))
cv2.imshow("Resized Kedi", resized_img)
# Resizing with fx and fy
# fx = 0.5, fy = 0.5 -> 50%
resized_img2 = cv2.resize(img, None, fx=0.5,fy=0.5)
cv2.imshow("Resized Kedi 2", resized_img2)
# Interpolation - İnterpolasyon

#Büyütme -> INTER_CUBIC -> 
big_img = cv2.resize(img, None, fx=1.2,fy=1.2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Big Kedi", big_img)

#Küçültme -> INTER_AREA
small_img = cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("Small Kedi", small_img)


# Rotate 
# 90,180,270
rotated_img = cv2.rotate(small_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("Rotated Kedi", rotated_img)
# 90-180-270 dışında ise
(h, w) = small_img.shape[:2]
center = (w//2, h//2)
matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1)

rotated_45 = cv2.warpAffine(small_img, matrix, (w, h))
cv2.imshow("Rotated 45 Kedi", rotated_45)
#

# Flip
flipped_img = cv2.flip(small_img, 1) # Aynalama
cv2.imshow("Flipped Kedi", flipped_img)

# Flip with direction
flipped_img2 = cv2.flip(small_img, 0) # Dikey aynalama
cv2.imshow("Flipped Kedi 2", flipped_img2)

flipped_img3 = cv2.flip(small_img, -1) # Hem Dikey hem yatay aynalama
cv2.imshow("Flipped Kedi 3", flipped_img3)
#

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("hsv_img.shape", hsv_img.shape)
print("hsv_img.ndim", hsv_img.ndim)
print("hsv_img[600,600]", hsv_img[600,600])
#cv2.imshow("HSV Kedi", hsv_img)


# Blurring - Smoothing
#(11,11) kernel size -> tek sayı olmak zorunda
blurred_img = cv2.GaussianBlur(small_img, (11,11), 0)
cv2.imshow("Blurred Kedi", blurred_img)
#
# 20:30

cv2.waitKey(0) # Kullanıcının herhangi bir tuşa basmasını bekle.
cv2.destroyAllWindows()


