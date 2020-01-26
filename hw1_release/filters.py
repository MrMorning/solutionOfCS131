"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    # print(Hk, Wk)
    ### YOUR CODE HERE
    x0 = Hk // 2
    y0 = Wk // 2
    # print(x0, y0)
    for m in range(Hi):
        for n in range(Wi):
            for i in range(-(Hk // 2), Hk // 2 + 1):
                for j in range(-(Wk // 2), Wk // 2 + 1):
                    try:
                        if m + i >= 0 and n + j >= 0 and x0 - i >= 0 and y0 - j >= 0:
                            out[m, n] += image[m + i, n + j] * kernel[x0 - i, y0 - j]
                        else:
                            # print('m=%d, n=%d, i=%d, j=%d' % (m, n, i, j), end="\n")
                            pass
                    except:
                        # print('m=%d, n=%d, i=%d, j=%d'%(m, n, i, j), end = "\n")
                        pass
    ### END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    left_zeroes = np.tile(np.zeros((H, 1)), pad_width)
    out = np.concatenate((left_zeroes, image, left_zeroes), axis=1)
    up_zeroes = np.tile(np.zeros((1, W + pad_width * 2)), (pad_height, 1))
    out = np.concatenate((up_zeroes, out, up_zeroes), axis=0)

    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image_with_padding = zero_pad(image, Hk // 2, Wk // 2)
    kernel_flipped = np.flip(kernel)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(kernel_flipped * image_with_padding[m: m + Hk, n: n + Wk])
    ### END YOUR CODE

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(g))
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    gg = g.copy()
    gg -= np.mean(gg)
    out = cross_correlation(f, gg)
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    image_with_padding = zero_pad(f, Hk // 2, Wk // 2)
    kernel = (g - np.mean(g)) / np.std(g)
    for m in range(Hi):
        for n in range(Wi):
            patch_image = np.copy(image_with_padding[m: m + Hk, n: n + Wk])
            patch_image = (patch_image - np.mean(patch_image)) / np.std(patch_image)
            out[m, n] = np.sum(kernel * patch_image)
    ### END YOUR CODE

    return out
