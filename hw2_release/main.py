# Setup
from __future__ import print_function

import matplotlib.pyplot as plt
import skimage.io as io

from edge import *

img = io.imread("cameraman.png", as_gray=True)
plt.subplot(121)
plt.imshow(img)
# plt.show()

plt.subplot(122)
plt.imshow(log(img))
plt.show()
