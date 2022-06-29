import os
import csv
from natsort import natsorted
from glob import glob

captions = ['CycleISP-RENOIR', 'CycleISP-SIDD-RENOIR', 'CycleISP-SIDD', 'MPRNet-RENOIR', 'MPRNet-SIDD-RENOIR', 'MPRNet-SIDD', 'Filtr dwustronny', 'Filtr Gaussowski', 'Filtr medianowy', 'Filtr Wienera']
datasets = ['SIDD', 'RENOIR']

for dataset in datasets:
    path = os.path.join('/content/output/real', dataset, '**', 'results.csv')
    files = natsorted(glob(path, recursive=True))
    for i in range(6):
      path = os.path.join('/content/output/real', dataset, str(i+1) + '.csv')
      results = open(path, 'w')
      writer = csv.writer(results)
      writer.writerow(['Algorytm', 'MSE', 'PSNR', 'SSIM'])
      for j, file in enumerate(files):
         with open(file) as fd:
          reader=csv.reader(fd)
          img, mse, psnr, ssim = [row for idx, row in enumerate(reader) if idx == i + 1][0]
          writer.writerow([captions[j], mse, psnr, ssim])
      print(f"Results saved at {path}")