import os
import csv
from natsort import natsorted
from glob import glob
from utils.names import getAlgorithmName

noises = ['gaussian', 'poisson', 'salt_pepper', 'speckle', 'real']

for noise in noises:
  basePath = os.path.join('/content/output/', noise)
  results = natsorted(glob(os.path.join(basePath, '**/results.csv'))
                    + glob(os.path.join(basePath, '**/**/results.csv')))
  path = os.path.join(basePath, 'avgResults.csv')
  avgResults = open(path, 'w')
  writer = csv.writer(avgResults)
  writer.writerow(['Algorytm', 'PSNR', 'SSIM'])
  
  for result in results:
    name = getAlgorithmName(result)
    avgPSNR = 0
    avgSSIM = 0 
    with open(result) as fd:
      reader = csv.reader(fd)
      for idx, row in enumerate(reader):
        if idx > 0:
          img, psnr, ssim = row
          avgPSNR += float(psnr)
          avgSSIM += float(ssim)
      writer.writerow([name, round(avgPSNR/6,3), round(avgSSIM/6,3)])
  print(f"Results saved at {path}")
