from scipy.signal import wiener
import numpy as np
import cv2

def gaussian_filter(img):
    blur = cv2.GaussianBlur(img,(5,5),1)
    return blur

def median_filter(img):
    median = cv2.medianBlur(img,5)    
    return median

def wiener_filter(img):
  result = np.zeros(img.shape)
  for i in range(3):
    result[:,:,i] = wiener(img[:,:,i], [5,5], 0)
  return result.astype(np.uint8)

def bilateral_filter(img):
    bilateral = cv2.bilateralFilter(img,9,75,75)    
    return bilateral

def getFilters():
    return [['gaussian', gaussian_filter], ['median', median_filter], ['wiener', wiener_filter], ['bilateral', bilateral_filter]]
