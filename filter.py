import cv2
import os
import csv
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from scipy.signal import wiener

def gaussian_filter(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur

def median_filter(img):
    median = cv2.medianBlur(img,5)    
    return median

def wiener_filter(img):
    filter = wiener(img, (5, 5), 0.7)
    return filter.astype(np.uint8)

def bilateral_filter(img):
    bilateral = cv2.bilateralFilter(img,9,75,75)    
    return bilateral
    
filters = [['gaussian', gaussian_filter], ['median', median_filter], ['wiener', wiener_filter], ['bilateral', bilateral_filter]]
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']

inp_dir = '/content/denoise/input'
noises = os.listdir(inp_dir)


for filter in filters: 
    name, add_filter = filter
    input = os.path.join('/content/output', name)

    for noise in noises:
        path = os.path.join(input, noise)
        results = open(os.path.join(path, 'results' + '.csv'), 'w')
        writer = csv.writer(results)
        writer.writerow(['MSE', 'PSNR', 'SSIM'])

        if not os.path.exists(path):
            os.makedirs(path)

        for image in images:
            ground = cv2.imread(os.path.join('/content/denoise/ground', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
            
            in_path = os.path.join('/content/denoise/input', noise, image + '.png')
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

            out_path = os.path.join(path, image + '.png')
            
            filter_img = add_filter(out_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(out_path, filter_img)
            mse = mean_squared_error(ground, filter_img)
            psnr = cv2.PSNR(ground, filter_img)
            (ssim, diff) = structural_similarity(ground, filter_img, full=True)
            writer.writerow([round(mse, 3), round(psnr, 3), round(ssim, 3)])
            
        print(f"Results saved at {path}")
