import matplotlib.pyplot as plt

from imageManip import *

image1_path = './len_std.jpg'
image2_path = './image1.jpg'


def display(img):
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


image1, image2 = np.array(load(image1_path))[:256, :256, :], np.array(load(image2_path))[:256, :256, :]
absOf1 = np.abs(np.fft.fft2(image1)).astype(np.complex128)
PhaOf2 = np.angle(np.fft.fft2(image2)).astype(np.complex128)
fftimage = np.real(np.fft.ifft2(absOf1 * np.exp(1j * PhaOf2)))
display(fftimage)

display(np.abs(np.fft.fft2(np.array(image))))
