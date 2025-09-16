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

cv2.imshow("Kedi BGR", img) #BGR
cv2.imshow("Kedi RGB", img2) #RGB
cv2.imshow("Kedi Grayscale", grayscale_img) #Grayscale
cv2.waitKey(0) # Kullanıcının herhangi bir tuşa basmasını bekle.
cv2.destroyAllWindows()

