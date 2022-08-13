import csv
import os
import numpy as np
import pandas as pd
from natsort import natsorted
from glob import glob

results = natsorted(glob(os.path.join('/content/output/', '**/avgResults.csv'), recursive=True)
                  + glob(os.path.join('/content/output/test/SIDD.csv'), recursive=True)
                  + glob(os.path.join('/content/output/test/RENOIR.csv'), recursive=True))

df = pd.read_csv(results[0]).drop(['PSNR','SSIM'], axis='columns')
algorithms = df.to_numpy().flatten()
size = len(results)
avgPSNR = np.zeros(len(algorithms), dtype = float)
avgSSIM = np.zeros(len(algorithms), dtype = float)

for result in results:
  with open(result) as fd:
    reader = csv.reader(fd)
    next(reader, None) 
    for idx, row in enumerate(reader):
      algorithm, psnr, ssim = row
      avgPSNR[idx] += float(psnr)
      avgSSIM[idx] += float(ssim)

avgPSNR /= size
avgSSIM /= size

output = '/content/output/sum'
os.makedirs(output, exist_ok=True)
path = os.path.join(output, 'avgResults.csv')
avgResults = open(path, 'w')
writer = csv.writer(avgResults)
writer.writerow(['Algorytm', 'PSNR', 'SSIM'])

for idx, algorithm in enumerate(algorithms):
  writer.writerow([algorithm, round(avgPSNR[idx],3), round(avgSSIM[idx],3)])

print(f"Results saved at {path}")
