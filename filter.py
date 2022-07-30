import cv2
import os
import numpy as np
from natsort import natsorted
from glob import glob
from scipy.signal import wiener
from utils.names import getTestDatasets

def gaussian_filter(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur

def median_filter(img):
    median = cv2.medianBlur(img,5)    
    return median

def wiener_filter(img):
  result = np.zeros(img.shape)
  for i in range(3):
    result[:,:,i] = wiener(img[:,:,i], [5,5], 0.7)
  return result

def bilateral_filter(img):
    bilateral = cv2.bilateralFilter(img,9,75,75)    
    return bilateral

filters = [['gaussian', gaussian_filter], ['median', median_filter], ['wiener', wiener_filter], ['bilateral', bilateral_filter]]
images = ['barbara', 'boats', 'cablecar', 'lena', 'mandril', 'peppers']
noise_datasets = getTestDatasets()

inp_dir = '/content/denoise/input'
noises = os.listdir(inp_dir)

def apply_filter(input_path, ground_path, output_path):
    ground = natsorted(glob(os.path.join(ground_path, '*.png'))
                    + glob(os.path.join(ground_path, '*.pgm'))
                    + glob(os.path.join(ground_path, '*.bmp')))
        
    input = natsorted(glob(os.path.join(input_path, '*.png'))
                    + glob(os.path.join(input_path, '*.pgm'))
                    + glob(os.path.join(input_path, '*.bmp')))     
                    
    for filter in filters: 
        name, add_filter = filter
        path = os.path.join(output_path, name)

        if not os.path.exists(path):
            os.makedirs(path)


        for file in input:
            img = cv2.imread(file)
            image = os.path.splitext(os.path.split(file)[-1])[0]
            out_path = os.path.join(path, image + '.png')
            
            filter_img = add_filter(img)
            cv2.imwrite(out_path, filter_img)
            
        print(f"Results saved at {path}")

for noise in noises:
    if noise == 'real':
        for dataset in noise_datasets:
            input_path = os.path.join('/content/denoise/input/real', dataset)
            ground_path = os.path.join('/content/denoise/ground', dataset)
            output_path = os.path.join('/content/output/real', dataset)
            apply_filter(input_path, ground_path, output_path)
    else:
        input_path = os.path.join('/content/denoise/input', noise)
        ground_path = '/content/denoise/ground'
        output_path = os.path.join('/content/output', noise)
        apply_filter(input_path, ground_path, output_path)
