import cv2

img = cv2.imread('lena.jpg',3)
cv2.namedWindow('image', cv2.WINDOW_NORMAL) # build window before load img, for autoisze window
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()