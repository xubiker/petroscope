from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import Iterable, List, Tuple


def transform_and_stitch(
    transforms: List[np.ndarray], img_paths: Iterable[Path]
) -> np.ndarray:
    """Stitches all the images into a single panorama
    using known transformations.

    Args:
        transforms : List of 3x3 homography matrices.
        img_paths : List of Path objects for each image of the panorama.

    Returns:
        np.ndarray: The resulting panorama image.
    """

    # Read and store each image
    pics = [cv2.imread(img_p).astype(np.float32) for img_p in img_paths]
    n = len(pics)

    # Initialize arrays to store original and transformed corner points
    all_corners = np.empty((n, 4, 3))
    for i in range(n):
        # Define corner points for each image
        all_corners[i] = [
            [0, 0, 1],
            [pics[i].shape[1], 0, 1],
            [pics[i].shape[1], pics[i].shape[0], 1],
            [0, pics[i].shape[0], 1],
        ]

    all_new_corners = np.empty((n, 4, 3))
    for i in range(n):
        # Apply homography transformations to each corner point
        all_new_corners[i] = [
            np.dot(transforms[i], corner) for corner in all_corners[i]
        ]

    # Reshape transformed corners for further processing
    all_new_corners = all_new_corners.reshape(-3, 3)
    x_news = all_new_corners[:, 0] / all_new_corners[:, 2]
    y_news = all_new_corners[:, 1] / all_new_corners[:, 2]

    # Determine min/max x and y coordinates for the panorama
    y_min = min(y_news)
    x_min = min(x_news)
    y_max = int(round(max(y_news)))
    x_max = int(round(max(x_news)))

    # Calculate shifts to adjust the panorama's origin
    x_shift = -min(x_min, 0)
    y_shift = -min(y_min, 0)
    T = np.array(
        [[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]], dtype="float32"
    )

    # Calculate new dimensions for the panorama
    x_min = int(round(x_min))
    y_min = int(round(y_min))
    height_new = y_max - y_min
    width_new = x_max - x_min
    size = (width_new, height_new)

    # Initialize the panorama using the first image
    panorama_ans = cv2.warpPerspective(
        src=pics[0],
        M=T @ transforms[0],
        dsize=size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(-1, -1, -1),
    )

    # Warp the remaining images into the panorama
    for i in range(1, n):
        cv2.warpPerspective(
            pics[i],
            T @ transforms[i],
            size,
            panorama_ans,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    return panorama_ans


def vec_to_homography(vec: np.ndarray, i: int, pivot: int) -> np.ndarray:
    """Extract a 3x3 homography matrix from a flattened vector.

    Args:
        vec (np.ndarray): Flattened vector of all homographies
            (except the pivot one, which is identical).

        i (int): The index of the homography to be extracted.

        pivot (int): The index of the pivot image.

    Returns:
        np.ndarray: The 3x3 homography matrix of the i-th image.
    """
    # If the index is the pivot, return the identity matrix
    if i == pivot:
        return np.eye(3)
    # Adjust index if it is greater than pivot
    elif i > pivot:
        i -= 1
    # Extract the 3x3 homography matrix from the vector
    H = vec[8 * i : 8 * (i + 1)]
    H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])
    return H


def homography_to_vec(Hs: List[np.ndarray], pivot: int) -> List[float]:
    """
    Flatten a list of 3x3 homography matrices into a single vector.

    Args:
        Hs (List[np.ndarray]): A list of all homography matrices.
        pivot (int): The index of the pivot image.

    Returns:
        List[float]: A flattened vector of all homography matrices
        (except the pivot).
    """
    n = len(Hs)
    vec = np.empty(8 * (n - 1))
    for i in range(n):
        if i == pivot:
            # Skip the pivot image
            continue
        elif i < pivot:
            # The homography matrix is placed at the position of the image
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * i : 8 * (i + 1)] = H
        else:
            # The homography matrix is placed at the position of the image
            # minus one (since the pivot image is skipped)
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * (i - 1) : 8 * i] = H
    return vec


def dist(X: List[float], inliers: List[np.ndarray], pivot: int) -> np.ndarray:
    """
    Calculate distances between the coordinates of all pairs of inliers
    in the transformed coordinate system.

    Args:
        X (List[float]): Flattened vector of all homographies
        (except the pivot one, which is identity).

        inliers (List[np.ndarray]): List of inliers, each inlier is
        an np.ndarray((i, j, x, y, xx, yy)), where i and j are the indices
        of the images corresponding to the inlier, (x, y) are the coordinates
        of the point on image i, and (xx, yy) are the coordinates of the
        point on image j.

        pivot (int): Index of the pivot image.

    Returns:
        np.ndarray: A vector of distances between the coordinates of all
        pairs of inliers in the transformed coordinates.
    """
    output = []  # Initialize the output list to store distances
    for i, j, x, y, xx, yy in inliers:
        # Get the homography matrices for images i and j
        Hi = vec_to_homography(X, i, pivot)
        Hj = vec_to_homography(X, j, pivot)

        # Transform the coordinates using the homography matrices
        first = np.dot(Hi, [x, y, 1])
        first /= first[2]  # Normalize to get the final coordinates
        second = np.dot(Hj, [xx, yy, 1])
        second /= second[2]  # Normalize to get the final coordinates
        output.append(first[0] - second[0])
        output.append(first[1] - second[1])

    return np.array(output)


def optimize(
    Hs: List[np.ndarray],
    inliers: List[np.ndarray],
    pivot: int,
) -> Tuple[List[np.ndarray], float, float]:
    """Global alignment using all inliers by adjusting all homography matrices.

    Args:
        Hs: a list of all homographies,

        inliers: list of inliers, each inlier is
        a np.ndarray((i, j, x, y, xx, yy)), where i and j are
        the indices of the images corresponding to the inlier,
        (x, y) are the coordinates of the point on image i,
        and (xx, yy) are the coordinates of the point on image j

        pivot: the number of the pivot image,

    Returns:
        a tuple of:
            * a list of new homographies,
            * the initial mean squared error,
            * the optimized mean squared error,
    """
    n = len(Hs)
    vec = homography_to_vec(Hs, pivot)
    norm = dist(vec, inliers, pivot)

    init_error = (norm**2).mean() ** 0.5
    res_lm = least_squares(
        dist, vec, method="lm", xtol=1e-6, ftol=1e-6, args=(inliers, pivot)
    )
    optim_error = (res_lm.fun**2).mean() ** 0.5
    new_vec = res_lm.x

    final_transforms = []
    for i in range(n):
        final_transforms.append(vec_to_homography(new_vec, i, pivot))
    return final_transforms, init_error, optim_error
