import cv2
import os
import csv
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from gaussian import gaussian_filter
from median import median_filter
from wiener import wiener_filter
from bilateral import bilateral_filter

filters = [['gaussian', gaussian_filter], ['median', median_filter], ['wiener', wiener_filter], ['bilateral', bilateral_filter]]
noises = ['salt_pepper', 'gaussian', 'poisson', 'speckle']
images = ['barbara', 'boat', 'chronometer', 'lena', 'mandril', 'peppers']
out = 'tex/img/teoria/filter'

for filter in filters:
    name = filter[0]
    add_filter = filter[1]
    results = open(os.path.join('code/filter/measure', name  + '.csv'), 'w')
    writer = csv.writer(results)

    for noise in noises:
        path = os.path.join(out, name, noise)

        if not os.path.exists(path):
            os.makedirs(path)

        for image in images:
            base = cv2.imread(os.path.join('code/noise/img', image + '.pgm'), cv2.IMREAD_GRAYSCALE)
            
            in_path = os.path.join('tex/img/teoria/noise', noise, image + '.png')
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

            filter_img = add_filter(img)
            out_path = os.path.join(path, image + '.png')
            cv2.imwrite(out_path, filter_img)

            mse = mean_squared_error(base, filter_img)
            psnr = cv2.PSNR(base, filter_img)
            (ssim, diff) = structural_similarity(base, filter_img, full=True)
            writer.writerow([round(mse, 3), round(psnr, 3), round(ssim, 3)])