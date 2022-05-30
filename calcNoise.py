import cv2
import os
import csv
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser(description='calc noise')
parser.add_argument('--name', default='MPRNet', type=str, help='Algorithm name')
parser.add_argument('--dataset', default='sidd', type=str, help='Dataset')
args = parser.parse_args()
name = args.name
dataset = args.dataset
input = os.path.join('/content/output', name, dataset)

noises = os.listdir('/content/denoise/input')
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']


for noise in noises:
    path = os.path.join(input, noise)
    results = open(os.path.join(path, 'results.csv'), 'w')
    writer = csv.writer(results)
    writer.writerow(['MSE', 'PSNR', 'SSIM'])

    if not os.path.exists(path):
        os.makedirs(path)

    for image in images:
        ground = cv2.imread(os.path.join('/content/denoise/ground', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
        
        in_path = os.path.join('/content/denoise/input', noise, image + '.png')
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

        out_path = os.path.join(path, image + '.png')
        
        denoiser_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        mse = mean_squared_error(ground, denoiser_img)
        psnr = cv2.PSNR(ground, denoiser_img)
        (ssim, diff) = structural_similarity(ground, denoiser_img, full=True)
        writer.writerow([round(mse, 3), round(psnr, 3), round(ssim, 3)])
        
    print(f"Results saved at {path}")
