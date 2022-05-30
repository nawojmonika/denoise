import cv2

def median_filter(img):
    median = cv2.medianBlur(img,5)    
    return median