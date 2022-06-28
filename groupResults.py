import os
import csv
from natsort import natsorted
from glob import glob

captions = ['CycleISP-RENOIR','CycleISP-SIDD', 'CycleISP-SIDD-RENOIR', 'MPRNet-RENOIR', 'MPRNet-SIDD', 'MPRNet-SIDD-RENOIR', 'Filtr dwustronny', 'Filtr Gaussowski', 'Filtr medianowy', 'Filtr Wienera']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']

noises = os.listdir('/content/output')
for noise in noises:
    path = os.path.join('/content/output', noise, '**', 'results.csv')
    files = natsorted(glob(path, recursive=True))
    for i, image in enumerate(images):
      path = os.path.join('/content/output', noise, image + '.csv')
      results = open(path, 'w')
      writer = csv.writer(results)
      writer.writerow(['Algorytm', 'MSE', 'PSNR', 'SSIM'])
      for j, file in enumerate(files):
         with open(file) as fd:
          reader=csv.reader(fd)
          img, mse, psnr, ssim = [row for idx, row in enumerate(reader) if idx == i + 1][0]
          writer.writerow([captions[j], mse, psnr, ssim])
      print(f"Results saved at {path}")