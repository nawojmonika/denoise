import os
import csv

algorithms = ['MPRNet', 'CycleISP']
noises = ['gaussian', 'poisson', 'salt_pepper', 'speckle']
out_path = '/content/output'

for i, noise in enumerate(noises):
    results = open(os.path.join(out_path, noise + '.csv'), 'w')
    writer = csv.writer(results)
    writer.writerow(['', 'MSE', 'PSNR', 'SSIM'])
    for algorithm in algorithms:
        path = os.path.join(out_path, algorithm)
        datasets = os.listdir(path)
        for dataset in datasets:
            path = os.path.join(out_path, algorithm, dataset, noise, 'results.csv')
            with open(path) as fd:
                reader=csv.reader(fd)
                mse, psnr, ssim =[row for idx, row in enumerate(reader) if idx == i + 1][0]
                writer.writerow([algorithm + '-' + dataset, mse, psnr, ssim])
                print(f"Results saved at {path}")
