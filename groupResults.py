import os
import csv
import pandas as pd
from natsort import natsorted
from glob import glob
from utils.names import getAlgorithmName, getTestDatasets

noises = os.listdir('/content/denoise/input')
datasets = getTestDatasets()

def writeResults(basePath, results):
  df = pd.read_csv(results[0]).drop(['PSNR','SSIM'], axis='columns')
  images = df.to_numpy().flatten()
  for i, image in enumerate(images):
      path = os.path.join(basePath, str(image) + '.csv')
      groupResults = open(path, 'w')
      writer = csv.writer(groupResults)
      writer.writerow(['Algorytm', 'PSNR', 'SSIM'])
      
      for result in results:
        name = getAlgorithmName(result)
        with open(result) as fd:
          reader=csv.reader(fd)
          img, psnr, ssim = [row for idx, row in enumerate(reader) if idx == i + 1][0]
          writer.writerow([name, psnr, ssim])
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