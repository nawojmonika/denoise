import cv2
import os
import csv
from skimage.metrics import structural_similarity
from natsort import natsorted
from glob import glob
from utils.names import getDatasets, getTestDatasets, getNetworkNames

noises = os.listdir('/content/denoise/input')
test_datasets = getTestDatasets()
datasets = getDatasets()
networks = getNetworkNames()

def calcResults(path, dataset = ''):
    results = open(os.path.join(path, 'results.csv'), 'w')
    writer = csv.writer(results)
    writer.writerow(['Obraz', 'PSNR', 'SSIM'])

    ground_path = os.path.join('/content/denoise/ground', dataset)
    ground = natsorted(glob(os.path.join(ground_path, '*.png'))
        + glob(os.path.join(ground_path, '*.bmp')))

    for ground_path in ground:
        name = ground_path.split('.')[0].split('/')[-1]
        out_path = os.path.join(path, name + '.png')
        ground_img = cv2.imread(ground_path)
        denoised_img = cv2.imread(out_path)
        psnr = cv2.PSNR(ground_img, denoised_img)
        (ssim, diff) = structural_similarity(ground_img, denoised_img, full=True, multichannel=True)
        writer.writerow([name, round(psnr, 3), round(ssim, 3)])

    print(f"Results saved at {path}/results.csv")
  

def writeResults(basePath, test_dataset = ''):
  algorithms = os.listdir(basePath)
  for algorithm in algorithms:
    if algorithm in networks:
      for dataset in datasets:
        path = os.path.join(basePath, algorithm, dataset)
        calcResults(path, test_dataset)
    else:
      path = os.path.join(basePath, algorithm)
      if os.path.isdir(path): 
        calcResults(path, test_dataset)

for noise in noises:
    if noise == 'real':
        for dataset in test_datasets:
            basePath = os.path.join('/content/output/real', dataset)
            results = natsorted(glob(os.path.join(basePath, '**/*.png'))
                        + glob(os.path.join(basePath, '**/**/*.png')))
            writeResults(basePath, dataset)
    else:
      basePath = os.path.join('/content/output', noise)
      writeResults(basePath)