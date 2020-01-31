"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

# from edge import gaussian_kernel
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks

from utils import *
from utils import pad


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)
    # print(img.shape)
    # print(dx.shape)
    dx2 = dx ** 2
    dy2 = dy ** 2
    dxdy = dx * dy
    ### YOUR CODE HERE
    sum_of_dx2 = convolve(dx2, window, mode='constant', cval=0.0)
    sum_of_dy2 = convolve(dy2, window, mode='constant', cval=0.0)
    sum_of_dxdy = convolve(dxdy, window, mode='constant', cval=0.0)
    response = sum_of_dx2 * sum_of_dy2 - sum_of_dxdy ** 2 - k * ((sum_of_dx2 + sum_of_dy2) ** 2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    feature = np.copy(patch)
    feature -= np.mean(feature)
    feature /= np.std(feature)
    # assert np.mean(feature) == 0 and np.std(feature) == 1
    feature = np.ravel(feature)
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    for i in range(N):
        mindist = np.min(dists[i])
        second_min_dist = np.sort(dists[i])[1]
        if (mindist / second_min_dist < threshold):
            matches.append((i, np.argmin(dists[i])))
    matches = np.array(matches)
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    print(N)
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N).astype(int)
    n_inliers = 0
    # RANSAC iteration start
    ### YOUR CODE HERE
    H = np.zeros((3, 3))
    for i in range(n_iters):
        selected_indices = np.random.choice(N, 8)
        selected_matched1 = matched1[selected_indices]
        selected_matched2 = matched2[selected_indices]
        assert selected_matched1.shape == (8, 3) and selected_matched2.shape == (8, 3)
        convert_H = np.linalg.lstsq(selected_matched2, selected_matched1, rcond=None)[0]
        # print(matched2.shape)
        # print(convert_H.shape)
        converted_matched1 = matched2 @ convert_H
        count_inliers = np.sum(np.sum(np.abs(converted_matched1 - matched1), axis=1) < threshold)
        if (count_inliers > n_inliers):
            max_inliers = np.arange(N)[np.sum(np.abs(converted_matched1 - matched1), axis=1) < threshold].astype(int)
            H = convert_H
            n_inliers = count_inliers

    ### END YOUR CODE
    # print(H)
    # print((max_inliers))
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    base_angle = 180 / n_bins
    block = np.zeros((0,))
    for i in range(rows):
        for j in range(cols):
            for x in range(G_cells[i, j].shape[0]):
                for y in range(G_cells[i, j].shape[1]):
                    angle_index = int(np.floor((theta_cells[i, j][x, y]) / base_angle))
                    if theta_cells[i, j][x, y] == 180:
                        angle_index = 8
                    cells[i, j, angle_index] += G_cells[i, j][x, y]
            block = np.concatenate((block, np.ravel(cells[i, j])))
    # block.reshape((((H*W*n_bins)/(M*N)), ))
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image
    # assert not np.any(img1_mask)
    # plt.imshow(img1_mask)
    # plt.show()
    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]
    # print(img1_warped[out_H // 2, -1])
    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    # print(img2_mask[out_H//2, :].shape)
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    weight_matrix_1 = np.ones((out_H, out_W))
    weight_matrix_1[:, left_margin:right_margin] = np.tile(np.linspace(1, 0, right_margin - left_margin), (out_H, 1))
    weight_matrix_1[:, right_margin:] = 0
    weight_matrix_2 = np.ones((out_H, out_W))
    weight_matrix_2[:, left_margin:right_margin] = np.tile(np.linspace(0, 1, right_margin - left_margin), (out_H, 1))
    weight_matrix_2[:, :left_margin] = 0
    merged = weight_matrix_1 * img1_warped + weight_matrix_2 * img2_warped
    # plt.imshow(img1_warped)
    # # plt.plot(np.array([left_margin for i in range(1000)]), np.linspace(0, out_H, 1000), 'r.')
    # # plt.plot(np.array([right_margin for i in range(1000)]), np.linspace(0, out_H, 1000), 'y.')
    # plt.show()
    # normalize_matrix = img1_mask * 1.0 + img2_mask
    # merged = merged / np.maximum(normalize_matrix, 1)
    # merged = weight_matrix_2 * img2_warped
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    convert_matrixs = [np.eye(3)]
    for i in range(1, len(imgs)):
        H, _ = ransac(keypoints[i - 1], keypoints[i], matches[i - 1], threshold=1)
        convert_matrix = H @ convert_matrixs[i - 1]
        convert_matrixs.append(convert_matrix)
    output_shape, offset = get_output_space(imgs[0], imgs[1:], convert_matrixs[1:])
    ## TODO: Here we use the left most image as reference
    panorama = warp_image(imgs[0], np.eye(3), output_shape, offset)
    panorama_mask = (panorama != -1)
    panorama[~panorama_mask] = 0
    for i in range(1, len(imgs)):
        # warped_imgs.append(warp_image(imgs[i], convert_matrixs[i], output_shape, offset))
        warped_image = warp_image(imgs[i], convert_matrixs[i], output_shape, offset)
        image_mask = (warped_image != -1)
        warped_image[~image_mask] = 0
        # plt.imshow(warped_image)
        # plt.show()
        panorama = linear_blend(panorama, warped_image)
        # plt.imshow(panorama)
        # plt.show()
    ### END YOUR CODE

    return panorama
