import cv2
import os
import csv
import numpy as np
from natsort import natsorted
from glob import glob
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
noise_datasets = ['RENOIR', 'SIDD']

inp_dir = '/content/denoise/input'
noises = os.listdir(inp_dir)

def apply_filter(input_path, ground_path, output_path):
    for filter in filters: 
        name, add_filter = filter
        
        path = os.path.join(output_path, name)

        if not os.path.exists(path):
            os.makedirs(path)

        results = open(os.path.join(path, 'results' + '.csv'), 'w')
        writer = csv.writer(results)
        writer.writerow(['', 'MSE', 'PSNR', 'SSIM'])

        ground = natsorted(glob(os.path.join(ground_path, '*.png'))
                    + glob(os.path.join(ground_path, '*.pgm')))
        
        input = natsorted(glob(os.path.join(input_path, '*.png'))
                    + glob(os.path.join(input_path, '*.pgm')))

        for i, file in enumerate(input):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            ground_img = cv2.imread(ground[i], cv2.IMREAD_GRAYSCALE)
            image = os.path.splitext(os.path.split(file)[-1])[0]
            out_path = os.path.join(path, image + '.png')
            
            filter_img = add_filter(img)
            cv2.imwrite(out_path, filter_img)
            mse = mean_squared_error(ground_img, filter_img)
            psnr = cv2.PSNR(ground_img, filter_img)
            (ssim, diff) = structural_similarity(ground_img, filter_img, full=True)
            writer.writerow([image, round(mse, 3), round(psnr, 3), round(ssim, 3)])
            
        print(f"Results saved at {path}")




for noise in noises:
    if noise == 'real':
        for dataset in noise_datasets:
            input_path = os.path.join('/content/denoise/input', noise, dataset)
            ground_path = os.path.join('/content/denoise/ground', dataset)
            output_path = os.path.join('/content/output', noise, dataset)
            apply_filter(input_path, ground_path, output_path)
    else:
        input_path = os.path.join('/content/denoise/input', noise)
        ground_path = 'content/denoise/ground'
        output_path = os.path.join('/content/output', noise, dataset)
        apply_filter(input_path, ground_path, output_path)
