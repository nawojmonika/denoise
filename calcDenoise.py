import cv2
import os
import csv
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser(description='calc denoise')
parser.add_argument('--name', default='MPRNet', type=str, help='Input images')
args = parser.parse_args()
name = args.name
out = '/content/denoise/output'

noises = ['salt_pepper', 'gaussian', 'poisson', 'speckle']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']

results = open(os.path.join(out+ '.csv'), 'w')
writer = csv.writer(results)

for noise in noises:
    path = os.path.join(out, noise)

    if not os.path.exists(path):
        os.makedirs(path)

    for image in images:
        base = cv2.imread(os.path.join('/content/denoise/ground', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
        
        in_path = os.path.join('/content/denoise/input', noise, image + '.png')
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

        out_path = os.path.join(path, image + '.png')
        
        denoiser_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        mse = mean_squared_error(base, denoiser_img)
        psnr = cv2.PSNR(base, denoiser_img)
        (ssim, diff) = structural_similarity(base, denoiser_img, full=True)
        writer.writerow([round(mse, 3), round(psnr, 3), round(ssim, 3)])