"""
CS131 - Computer Vision: Foundations and Applications
Assignment 5
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/25/2018
Python Version: 3.5+
"""

import numpy as np
import scipy
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    last_centers = centers
    for n in range(num_iters):
        ### YOUR CODE HERE
        means = np.zeros((k, D))
        tot = np.zeros((k))
        assignments = np.zeros(N, dtype=np.uint32)
        for i in range(len(idxs)):
            assignments[idxs[i]] = i+1
            means[i] += features[idxs[i]]
            tot[i]+=1
        for i in range(N):
            if(assignments[i]):
                continue
            nearest_center_id = idxs[0]
            mindist = np.linalg.norm(features[i] - centers[0])
            # for center in centers[1:]:
            for j in range(k-1):
                center = centers[j+1]
                current_dist = np.linalg.norm(center - features[i])
                if(current_dist < mindist):
                    mindist = current_dist
                    nearest_center_id = idxs[j+1]
            assignments[i] = assignments[nearest_center_id]
            means[assignments[i]-1] += features[i]
            tot[assignments[i]-1] += 1
        means = means / tot.reshape((4,1))

        idxs = np.zeros((k)).astype(int)
        mindist_of_cluster = np.zeros((k))
        inf = 1000000000
        mindist_of_cluster += inf
        for i in range(N):
            current_dist = np.linalg.norm(features[i] - means[assignments[i]-1])
            if(current_dist < mindist_of_cluster[assignments[i]-1]):
                mindist_of_cluster[assignments[i]-1] = current_dist
                idxs[assignments[i]-1] = i
        last_centers = centers
        centers = features[idxs]
        if(np.allclose(centers, last_centers)):
            break
    assignments -= 1
        ### END YOUR CODE
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        measure_matrix = np.tile(features.reshape(N, 1, D), (1, k, 1))
        measure_matrix = np.sum((measure_matrix - centers.reshape((1, k, D))) ** 2, axis = 2)
        assignments = np.argmin(measure_matrix, axis=1)
        mean = np.zeros((k, D))
        for i in range(k):
            mean[i] = np.mean(features[assignments == i], axis=0)
        another_measure_matrix = np.tile(features.reshape(1, N, D), (k, 1, 1))
        another_measure_matrix = np.sum((another_measure_matrix - mean.reshape((k, 1, D))) ** 2, axis=2)
        last_centers = centers
        idxs = np.argmin(another_measure_matrix, axis=1)
        centers = features[idxs]
        if (np.allclose(centers, last_centers)):
            break
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE
        infty = 100000;
        distance_matrix = np.sum((centers.reshape((1, N, D)) - centers.reshape((N, 1, D))) ** 2, axis=2)
        np.fill_diagonal(distance_matrix, infty)
        distance_matrix[(centers == infty)[:, 0], :] = infty
        distance_matrix[:, (centers == infty)[:, 0]] = infty
        x, y = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        if x > y:
            x, y = y, x
        centers[x] = np.mean(features[np.logical_or(assignments == x, assignments == y)], axis=0)
        centers[y] = infty
        assignments[assignments==y] = x
        n_clusters -= 1
    id = 0
    for i in range(N):
        if(centers[i, 0] != infty):
            assignments[assignments == i] = id
            id += 1
    assert id == k
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape((H * W, C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    spatial = np.mgrid[0:H, 0:W].astype(float)
    color -= np.mean(color, axis=(0, 1))
    color /= np.std(color, axis=(0, 1))
    spatial[0] -= np.mean(spatial[0])
    spatial[1] -= np.mean(spatial[1])
    spatial[0] /= np.std(spatial[0])
    spatial[1] /= np.std(spatial[1])
    features = np.concatenate((color, np.dstack((spatial[0], spatial[1]))), axis=2)
    features = features.reshape((H*W, C+2))
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    accuracy = np.mean(mask_gt == mask)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
