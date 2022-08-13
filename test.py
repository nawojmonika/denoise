import cv2
import os
import csv
import torch
import numpy as np
import scipy.io as sio
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from runpy import run_path
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity
from utils.filters import getFilters
from utils.names import getFilterName, getNetworkNames, getDatasets, getTestDatasets
from utils.models import load_checkpoint

models =  getNetworkNames()
datasets = getDatasets()
test_datasets = getTestDatasets()
filters = getFilters()
img_multiple_of = 8


path = '/content/output/test'
if not os.path.exists(path):
  os.makedirs(path)

for test_dataset in test_datasets:
    
    results = open(os.path.join(path, test_dataset + '.csv'), 'w')
    writer = csv.writer(results)
    writer.writerow(['Algorytm', 'PSNR', 'SSIM'])

    # Process data
    basepath = os.path.join('/content/test', test_dataset)
    noisy = sio.loadmat(os.path.join(basepath, 'testNoisyBlocksSrgb.mat'))
    gt = sio.loadmat(os.path.join(basepath, 'testGtBlocksSrgb.mat'))

    Inoisy = np.float32(np.array(noisy['ValidationNoisyBlocksSrgb']))
    Inoisy /=255.

    gt = np.float32(np.array(gt['ValidationGtBlocksSrgb']))
    gt /=255.


    for name in models:
        load_file = run_path(os.path.join('/content/denoise', name + '.py'))
        model = load_file[name]()
        model.cuda()

        for dataset in datasets:
            weights = os.path.join('/content/models', name + '_' + dataset + '.pth')
            load_checkpoint(model, weights)
            model.eval()
            total_psnr = 0;
            total_ssim = 0;
            with torch.no_grad():
                for i in tqdm(range(40)):
                    for k in range(32):
                        noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                        noisy_img = torchvision.utils.make_grid(noisy_patch.cpu(), nrow=5).permute(1, 2, 0)
                        noisy_img = img_as_ubyte(noisy_img)

                        gt_patch = torch.from_numpy(gt[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                        gt_img = torchvision.utils.make_grid(gt_patch.cpu(), nrow=5).permute(1, 2, 0)
                        gt_img = img_as_ubyte(gt_img)

                        input_ = TF.to_tensor(noisy_img).unsqueeze(0).cuda()

                        # Pad the input if not_multiple_of 8
                        h,w = input_.shape[2], input_.shape[3]
                        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
                        padh = H-h if h%img_multiple_of!=0 else 0
                        padw = W-w if w%img_multiple_of!=0 else 0
                        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
                        restored = model(input_)

                        if name == 'MPRNet':
                            restored = restored[0]
                        restored = torch.clamp(restored, 0, 1)

                        # Unpad the output
                        restored = restored[:,:,:h,:w]

                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = img_as_ubyte(restored[0])

                        total_psnr += cv2.PSNR(gt_img, restored)
                        (ssim, diff) = structural_similarity(gt_img, restored, full=True, multichannel=True)
                        total_ssim +=ssim
            
            network_name = name + '-' + dataset.upper();
            qm_psnr = total_psnr / (40*32);
            qm_ssim = total_ssim / (40*32);
            writer.writerow([network_name, round(qm_psnr, 3), round(qm_ssim, 3)])
    
    for filter in filters:
        total_psnr = 0;
        total_ssim = 0;
        name, add_filter = filter
        with torch.no_grad():
            for i in tqdm(range(40)):
                for k in range(32):
                    noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                    noisy_img = torchvision.utils.make_grid(noisy_patch.cpu(), nrow=5).permute(1, 2, 0)
                    noisy_img = img_as_ubyte(noisy_img)

                    gt_patch = torch.from_numpy(gt[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                    gt_img = torchvision.utils.make_grid(gt_patch.cpu(), nrow=5).permute(1, 2, 0)
                    gt_img = img_as_ubyte(gt_img)

                    denoised_img = add_filter(noisy_img)
                    total_psnr += cv2.PSNR(gt_img, denoised_img)
                    (ssim, diff) = structural_similarity(gt_img, denoised_img, full=True, multichannel=True)
                    total_ssim +=ssim

        qm_psnr = total_psnr / (40*32);
        qm_ssim = total_ssim / (40*32);
        writer.writerow([getFilterName(name), round(qm_psnr, 3), round(qm_ssim, 3)])
    
    print(f"Results saved at {path}/results.csv")