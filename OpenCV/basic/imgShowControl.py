import cv2

img = cv2.imread('lena.jpg', 3)
cv2.imshow('image', img)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
elif key == ord('s'):
    cv2.imwrite('lena2.jpg', img)
    cv2.destroyAllWindows()
