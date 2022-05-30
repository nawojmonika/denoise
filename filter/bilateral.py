import cv2

def bilateral_filter(img):
    bilateral = cv2.bilateralFilter(img,9,75,75)    
    return bilateral