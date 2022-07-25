import os
import csv
from natsort import natsorted
from glob import glob
from utils.names import getAlgorithmName, getTestDatasets

noises = os.listdir('/content/denoise/input')
datasets = getTestDatasets()

def writeResults(basePath, results):
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


for noise in noises:
  if noise == 'real':
    for dataset in datasets:
      basePath = os.path.join('/content/output/', noise, dataset)
      results = natsorted(glob(os.path.join(basePath, '**/results.csv'), recursive=True))
      writeResults(basePath, results)
  else:
    basePath = os.path.join('/content/output/', noise)
    results = natsorted(glob(os.path.join(basePath, '**/results.csv'), recursive=True))
    writeResults(basePath, results)
