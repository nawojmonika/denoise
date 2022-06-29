import cv2
import os
import csv
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from natsort import natsorted
from glob import glob
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='calc noise')
parser.add_argument('--name', default='MPRNet', type=str, help='Algorithm name')
parser.add_argument('--dataset', default='sidd', type=str, help='Dataset')
args = parser.parse_args()
name = args.name
dataset = args.dataset

noises = os.listdir('/content/denoise/input')
noise_datasets = ['RENOIR', 'SIDD']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']

for noise in noises:
    path = os.path.join('/content/output', noise, name, dataset)
    results = open(os.path.join(path, 'results.csv'), 'w')
    writer = csv.writer(results)
    writer.writerow(['', 'MSE', 'PSNR', 'SSIM'])

    if not os.path.exists(path):
        os.makedirs(path)

    if noise == 'real':
        for dataset in noise_datasets:
            ground_path = os.path.join('/content/denoise/ground', dataset)
            ground = natsorted(glob(os.path.join(ground_path, '*.png'))
                    + glob(os.path.join(ground_path, '*.pgm')))

            for i, ground_path in enumerate(ground):
                ground_img = cv2.imread(ground_path, cv2.IMREAD_GRAYSCALE)
                out_path = os.path.join(path, str(i+1) + '.png')
                # print(ground_path)
                # print(out_path)
        
                denoised_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
                mse = mean_squared_error(ground_img, denoised_img)
                psnr = cv2.PSNR(ground_img, denoised_img)
                (ssim, diff) = structural_similarity(ground_img, denoised_img, full=True)
                writer.writerow([i+1, round(mse, 3), round(psnr, 3), round(ssim, 3)])

    else: 
        for image in images:
            ground = cv2.imread(os.path.join('/content/denoise/ground', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
            out_path = os.path.join(path, image + '.png')
            
            denoiser_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            mse = mean_squared_error(ground, denoiser_img)
            psnr = cv2.PSNR(ground, denoiser_img)
            (ssim, diff) = structural_similarity(ground, denoiser_img, full=True)
            writer.writerow([image, round(mse, 3), round(psnr, 3), round(ssim, 3)])
        
    print(f"Results saved at {path}")