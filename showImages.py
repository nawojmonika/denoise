import cv2
import argparse
from natsort import natsorted
from glob import glob
from matplotlib import pyplot as plt
from numpy import number

captions = ['CycleISP-DND', 'CycleISP-SIDD', 'MPRNet-SIDD', 'Filtr dwustronny', 'Filtr Gaussowski', 'Filtr medianowy', 'Filtr Wienera']

parser = argparse.ArgumentParser(description='Show result images')
parser.add_argument('--path', type=str, help='Path to images')
parser.add_argument('--rows', default=2, type=number, help='Number of rows')
parser.add_argument('--cols', default=3, type=number, help='Number of columns')
parser.add_argument('--showCaptions', default=None, type=str, help='Show captions')
args = parser.parse_args()

path = args.path
rows = args.rows
cols = args.cols
showCaptions = args.showCaptions

fig = plt.figure(figsize=(10, 7))
files = natsorted(glob(path, recursive=True))

for i, file in enumerate(files):
    fig.add_subplot(rows, cols, i+1)
    image = cv2.imread(file)
    plt.imshow(image)
    plt.axis('off')
    if showCaptions == False:
        plt.title(i)
    else:
        plt.title(captions[i])