# Setup
from __future__ import print_function

from itertools import product
from os import listdir

import numpy as np
import skimage.io as io

from edge import canny

# Define parameters to test
sigmas = list(np.arange(0.5, 2., 0.1))
highs = list(np.arange(0.01, 0.1, 0.01))
lows = list(np.arange(0.01, 0.1, 0.01))

maxf1 = 0.1494
ans = (1.4, 0.03, 0.02)
for sigma, high, low in product(sigmas, highs, lows):
    if (high <= low):
        continue
    print("sigma={}, high={}, low={}".format(sigma, high, low))
    n_detected = 0.0
    n_gt = 0.0
    n_correct = 0.0

    for img_file in listdir('images/objects'):
        img = io.imread('images/objects/' + img_file, as_gray=True)
        gt = io.imread('images/gt/' + img_file + '.gtf.pgm', as_gray=True)

        mask = (gt != 5)  # 'don't' care region
        gt = (gt == 0)  # binary image of GT edges

        edges = canny(img, kernel_size=5, sigma=sigma, high=high, low=low)
        edges = edges * mask

        n_detected += np.sum(edges)
        n_gt += np.sum(gt)
        n_correct += np.sum(edges * gt)

    p_total = n_correct / n_detected
    r_total = n_correct / n_gt
    f1 = 2 * (p_total * r_total) / (p_total + r_total)
    print('Total precision={:.4f}, Total recall={:.4f}'.format(p_total, r_total))
    print('F1 score={:.4f}'.format(f1))
    if (f1 > maxf1):
        maxf1 = f1
        ans = sigma, high, low

print(maxf1)
print(ans)
