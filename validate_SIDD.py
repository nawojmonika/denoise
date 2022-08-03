import numpy as np
import os
from tqdm import tqdm
import torch
import scipy.io as sio
import torchvision
from filter import filters
print(filters)
# from utils.names import getTestDatasets, getDatasets, getNetworkNames

# def restore(noisy, gt):


# Process data
# basepath = '/content/validation/SIDD'
# noisy = sio.loadmat(os.path.join(basepath, 'ValidationNoisyBlocksSrgb.mat'))
# gt = sio.loadmat(os.path.join(basepath, 'ValidationGtBlocksSrgb.mat'))

# Inoisy = np.float32(np.array(noisy['ValidationNoisyBlocksSrgb']))
# Inoisy /=255.

# gt = np.float32(np.array(gt['ValidationGtBlocksSrgb']))
# gt /=255.

# total_psnr = 0;
# total_ssim = 0;

# with torch.no_grad():
#     for i in tqdm(range(40)):
#         for k in range(32):
#             noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
#             noisy_img = torchvision.utils.make_grid(noisy_patch.cpu(), nrow=5).permute(1, 2, 0)
#             gt_patch = torch.from_numpy(gt[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
#             gt_img = torchvision.utils.make_grid(gt_patch.cpu(), nrow=5).permute(1, 2, 0)
