import cv2
import os
import csv
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--name', default='MPRNet', type=str, help='Input images')
args = parser.parse_args()
name = args.name

noises = ['salt_pepper', 'gaussian', 'poisson', 'speckle']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']
out = 'tex/img/denoise'

results = open(os.path.join('/content/out/', name + '.csv'), 'w')
writer = csv.writer(results)

for noise in noises:
    path = os.path.join(out, name, noise)

    if not os.path.exists(path):
        os.makedirs(path)

    for image in images:
        base = cv2.imread(os.path.join('/content/drive/MyDrive/noise/base', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
        
        in_path = os.path.join('/content/drive/MyDrive/noise', noise, image + '.png')
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

        out_path = os.path.join(path, image + '.png')
        
        denoiser_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        mse = mean_squared_error(base, denoiser_img)
        psnr = cv2.PSNR(base, denoiser_img)
        (ssim, diff) = structural_similarity(base, denoiser_img, full=True)
        writer.writerow([round(mse, 3), round(psnr, 3), round(ssim, 3)])