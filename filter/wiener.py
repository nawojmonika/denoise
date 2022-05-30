from scipy.signal import wiener
import numpy as np

def wiener_filter(img):
    filter = wiener(img, (5, 5), 0.7)
    return filter.astype(np.uint8)