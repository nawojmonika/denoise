import cv2
import os
import csv
import torch
import numpy as np
import scipy.io as sio
import torchvision
from tqdm import tqdm
from skimage.metrics import structural_similarity
from utils.filters import getFilters
from utils.names import getFilterName

filters = getFilters()
path = '/content/validation/SIDD'
results = open(os.path.join(path, 'results.csv'), 'w')
writer = csv.writer(results)
writer.writerow(['', 'PSNR', 'SSIM'])

# Process data
basepath = '/content/validation/SIDD'
noisy = sio.loadmat(os.path.join(basepath, 'ValidationNoisyBlocksSrgb.mat'))
gt = sio.loadmat(os.path.join(basepath, 'ValidationGtBlocksSrgb.mat'))

Inoisy = np.float32(np.array(noisy['ValidationNoisyBlocksSrgb']))
Inoisy /=255.

gt = np.float32(np.array(gt['ValidationGtBlocksSrgb']))
gt /=255.

for filter in filters:
    total_psnr = 0;
    total_ssim = 0;
    name, add_filter = filter

    with torch.no_grad():
        for i in tqdm(range(40)):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                noisy_img = torchvision.utils.make_grid(noisy_patch.cpu(), nrow=5).permute(1, 2, 0)
                gt_patch = torch.from_numpy(gt[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                gt_img = torchvision.utils.make_grid(gt_patch.cpu(), nrow=5).permute(1, 2, 0)
                
                denoised_img = add_filter(noisy_img)
                total_psnr += cv2.PSNR(gt_img, denoised_img)
                (ssim, diff) = structural_similarity(gt_img, denoised_img, full=True, multichannel=True)
                total_ssim +=ssim


    qm_psnr = total_psnr / (40*32);
    qm_ssim = total_ssim / (40*32);
    writer.writerow([getFilterName(name), round(qm_psnr, 3), round(qm_ssim, 3)])

print(f"Results saved at {path}/results.csv")
