import cv2
import os
from natsort import natsorted
from glob import glob
from utils.names import getTestDatasets
from utils.filters import getFilters

images = ['barbara', 'boats', 'cablecar', 'lena', 'mandril', 'peppers']
noise_datasets = getTestDatasets()
filters = getFilters()

inp_dir = '/content/denoise/input'
noises = os.listdir(inp_dir)

def apply_filter(input_path, output_path):        
    input = natsorted(glob(os.path.join(input_path, '*.png'))
                    + glob(os.path.join(input_path, '*.pgm'))
                    + glob(os.path.join(input_path, '*.bmp')))     
                    
    for filter in filters: 
        name, add_filter = filter
        path = os.path.join(output_path, name)

        if not os.path.exists(path):
            os.makedirs(path)


        for file in input:
            img = cv2.imread(file)
            image = os.path.splitext(os.path.split(file)[-1])[0]
            out_path = os.path.join(path, image + '.png')
            
            filter_img = add_filter(img)
            cv2.imwrite(out_path, filter_img)
            
        print(f"Results saved at {path}")

for noise in noises:
    if noise == 'real':
        for dataset in noise_datasets:
            input_path = os.path.join('/content/denoise/input/real', dataset)
            output_path = os.path.join('/content/output/real', dataset)
            apply_filter(input_path, output_path)
    else:
        input_path = os.path.join('/content/denoise/input', noise)
        output_path = os.path.join('/content/output', noise)
        apply_filter(input_path, output_path)
