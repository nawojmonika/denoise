import cv2

def gaussian_filter(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur