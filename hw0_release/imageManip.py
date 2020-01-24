import numpy as np
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * image * image
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2grey(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    print(image.shape)
    ### YOUR CODE HERE
    dic = {'R':0, 'G':1, 'B':2}
    #mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    rgb = np.array(image).T
    rgb[dic[channel]] *= 0
    out = rgb.T
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    dic = {'L':0, 'A':1, 'B':2}
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out = np.array(lab.copy()) * mat[dic[channel]]
    out = color.lab2rgb(out.copy())
    # a = np.ones(10).reshape((3, 4))
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    lis = ['H', 'S', 'V']
    dic = {x: i for i, x in enumerate(lis)}
    mat = np.eye(3)
    onl = hsv * mat[dic[channel]]
    out = color.hsv2rgb(onl)
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    left_part = np.array(rgb_exclusion(image1, channel1))
    right_part = np.array(rgb_exclusion(image2, channel2))
    w1 = int(left_part.shape[1])
    w2 = int(right_part.shape[1])
    out = np.concatenate((left_part[:, :w1 // 2, :], right_part[:, -w2 // 2:, :]), axis=1)
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    image_ndarray = np.array(image)
    image_width = image_ndarray.shape[1]
    image_height = image_ndarray.shape[0]
    left_top = rgb_exclusion(image_ndarray[:image_height // 2, :image_width // 2, :], 'R')
    right_top = dim_image(image_ndarray[:image_height // 2, -image_width // 2:, :])
    left_bottom = np.sqrt(image_ndarray[-image_height // 2:, :image_width // 2, :])
    right_bottom = rgb_exclusion(image_ndarray[-image_height // 2:, -image_width // 2:, :], 'R')
    out = np.concatenate((np.concatenate((left_top, right_top), axis=1),
                          np.concatenate((left_bottom, right_bottom), axis=1)), axis=0)
    ### END YOUR CODE

    return out
