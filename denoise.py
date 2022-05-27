import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = '/content/denoise/input'
out_dir = '/content/output/MPRNet'
noises = os.listdir(inp_dir)

# Load corresponding model architecture and weights
load_file = run_path("/content/denoise/MPRNet.py")
model = load_file['MPRNet']()
model.cuda()

weights = os.path.join("/content/drive/MyDrive/models", "MPRNet.pth")
load_checkpoint(model, weights)
model.eval()

img_multiple_of = 8

for noise in noises:
    input_path = os.path.join(inp_dir, noise)
    output_path = os.path.join(out_dir, noise)
    os.makedirs(output_path, exist_ok=True)

    files = natsorted(glob(os.path.join(input_path, '*.jpg'))
                    + glob(os.path.join(input_path, '*.JPG'))
                    + glob(os.path.join(input_path, '*.png'))
                    + glob(os.path.join(input_path, '*.PNG')))

    if len(files) == 0:
        raise Exception(f"No files found at {input_path}")


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
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(output_path, f+'.png')), restored)

    print(f"Files saved at {output_path}")
