from scipy.signal import wiener
import numpy as np
import cv2
import math

def gaussian_filter(img):
    blur = cv2.GaussianBlur(img,(5,5),1)
    return blur

def median_filter(img):
    median = cv2.medianBlur(img,5)    
    return median

def wiener_filter(img):
  result = np.zeros(img.shape)
  for i in range(3):
    result[:,:,i] = wiener(img[:,:,i], noise=0)
  return result.astype(np.uint8)

def bilateral_filter(img):
    h, w, c = img.shape
    sigma_r = np.mean(cv2.Scharr(img, -1, 0, 1))
    sigma_s = math.sqrt(pow(w,2) + pow(h,2)) * 0.02
    bilateral = cv2.bilateralFilter(img,-1,sigma_r,sigma_s)    
    return bilateral

def getFilters():
    return [['bilateral', bilateral_filter], ['gaussian', gaussian_filter], ['median', median_filter], ['wiener', wiener_filter]]
