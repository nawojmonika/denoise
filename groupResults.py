import os
import csv
from natsort import natsorted
from glob import glob

captions = ['CycleISP-RENOIR', 'CycleISP-SIDD-RENOIR', 'CycleISP-SIDD', 'MPRNet-RENOIR', 'MPRNet-SIDD-RENOIR', 'MPRNet-SIDD', 'Filtr dwustronny', 'Filtr Gaussowski', 'Filtr medianowy', 'Filtr Wienera']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']
noises = ['gaussian', 'poisson', 'salt_pepper', 'speckle']

for noise in noises:
    path = os.path.join('/content/output', noise, '**', 'results.csv')
    files = natsorted(glob(path, recursive=True))
    for i, image in enumerate(images):
      path = os.path.join('/content/output', noise, image + '.csv')
      results = open(path, 'w')
      writer = csv.writer(results)
      writer.writerow(['Algorytm', 'PSNR', 'SSIM'])
      for j, file in enumerate(files):
         with open(file) as fd:
          reader=csv.reader(fd)
          img, psnr, ssim = [row for idx, row in enumerate(reader) if idx == i + 1][0]
          writer.writerow([captions[j], psnr, ssim])
      print(f"Results saved at {path}")