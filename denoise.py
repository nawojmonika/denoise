import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from utils.names import getTestDatasets, getDatasets, getNetworkNames
from utils.models import load_checkpoint

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def denoise(input_path, output_path):
    files = natsorted(glob(os.path.join(input_path, '*.jpg'))
                    + glob(os.path.join(input_path, '*.JPG'))
                    + glob(os.path.join(input_path, '*.pgm'))
                    + glob(os.path.join(input_path, '*.bmp'))
                    + glob(os.path.join(input_path, '*.png'))
                    + glob(os.path.join(input_path, '*.PNG')))

    if len(files) == 0:
        raise Exception(f'No files found at {input_path}')


    for file_ in files:
        img = Image.open(file_).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = model(input_)

            if name == 'MPRNet':
                restored = restored[0]
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:h,:w]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            f = os.path.splitext(os.path.split(file_)[-1])[0]
            save_img((os.path.join(output_path, f+'.png')), restored)

    print(f'Files saved at {output_path}')

inp_dir = '/content/denoise/input'
noises = os.listdir(inp_dir)
models =  getNetworkNames()
datasets = getDatasets()
noise_datasets = getTestDatasets()
img_multiple_of = 8

# Load corresponding model architecture and weights
for name in models:
    load_file = run_path(os.path.join('/content/denoise', name + '.py'))
    model = load_file[name]()
    model.cuda()

    for dataset in datasets:
        weights = os.path.join('/content/models', name + '_' + dataset + '.pth')
        load_checkpoint(model, weights)
        model.eval()

        for noise in noises:
            if noise == 'real':
                for real_dataset in noise_datasets:
                    input_path = os.path.join(inp_dir, noise, real_dataset)
                    output_path = os.path.join('/content/output/real', real_dataset, name, dataset)
                    os.makedirs(output_path, exist_ok=True)
                    denoise(input_path, output_path)

            else:
                input_path = os.path.join(inp_dir, noise)
                output_path = os.path.join('/content/output', noise, name, dataset)
                os.makedirs(output_path, exist_ok=True)
                denoise(input_path, output_path)