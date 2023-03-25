import numpy as np
import cv2 as cv
import imutil as imu
import itertools
from scipy import spatial as scp
import random as rng


seed = 42
rng.seed(seed)
np.random.seed(seed)


def get_point_kernel(radius):
    kernel = np.zeros((radius * 2 + 1,) * 2, bool)
    kernel[radius, radius] = True
    return kernel


def calculate_point_score(img, alpha, binary_mask):
    # calculate corner score
    corner_val = cv.cornerMinEigenVal(
        imu.gray_mean(img.copy()).astype(np.uint8) * alpha, 3
    )
    corner_val -= np.amin(corner_val)
    corner_val /= np.amax(corner_val)
    corner_val = np.where(
        binary_mask, corner_val, np.random.uniform(0, 1, (corner_val.shape))
    )

    # calculate offset to ensure points at image boundary
    offset = np.zeros_like(corner_val)
    # prioritize points on opaque areas over transparent
    offset = np.where(binary_mask, 1, offset)

    coords = (0, -1)
    # edges
    for k in coords:
        offset[:, k] = 2
        offset[k, :] = 2

    # corners
    for k in itertools.product(coords, coords):
        offset[k] = 3

    # imu.imshow(imu.to_uint(offset / 3))

    return corner_val + offset * 2


def iterate_kernel(argsorted_score, point_mask, kernel):
    kernel_size = kernel.shape[0]
    half = kernel_size // 2
    for x, y in argsorted_score:
        if not point_mask[x + half, y + half]:
            continue
        point_mask[x : x + kernel_size, y : y + kernel_size] = kernel


def narrow_down_points(score, size):
    kernel_size = size * 2 + 1
    kernel = get_point_kernel(size)
    res = np.array(score.shape, np.int32)

    # sort corner score in descending order
    score_argsorted = np.unravel_index(np.argsort(score, axis=None), score.shape)
    score_argsorted = np.stack(score_argsorted, 1).astype(np.uint32)[::-1]

    # create a point mask with padding to account for kernel size
    point_mask = np.ones(res + size * 2, bool)

    iterate_kernel(score_argsorted, point_mask, kernel)

    imu.imshow(point_mask.astype(np.uint8) * 255)

    return np.stack(np.where(point_mask[size:, size:]), 1, dtype=np.int32)


def crop_to_divisable(img, denominator):
    # crop image to make it divisable by denominator
    res = np.array(img.shape, np.int32)

    cropped = np.zeros(
        (np.ceil(res / denominator) * denominator).astype(np.int32), img.dtype
    )
    cropped[: res[0], : res[1]] = img
    return cropped


def fold(arr, size):
    folded = arr.reshape(-1, size, *arr.shape[1:])
    folded = np.moveaxis(folded, 1, 2)
    return folded.reshape(folded.shape[0], -1, size**2, folded.shape[3])


def narrow_down_points_cell_based(score, size, kernel_size=2):
    kernel = get_point_kernel(kernel_size)

    # crop for downsampling
    img = crop_to_divisable(score, size)

    # create an index array to map from folded array to original
    indices = np.moveaxis(np.indices(img.shape, np.int32), 0, -1)

    # divide into cells, similar to poisson disk sampling
    indices_folded = fold(indices, size)
    img_folded = img[indices_folded[..., 0], indices_folded[..., 1]]

    # indices of max value in each cell
    argmax = np.argmax(img_folded, -1)
    # max value in each cell
    amax = np.take_along_axis(img_folded, argmax[..., None], 2)[..., 0]
    # index of max value in each cell
    amax_indices = np.take_along_axis(indices_folded, argmax[..., None, None], 2)[
        :, :, 0
    ]

    # sort score in descending order
    score_argsorted = np.unravel_index(np.argsort(amax, axis=None), amax.shape)
    score_argsorted = np.stack(score_argsorted, 1).astype(np.uint32)[::-1]

    # create a point mask with padding to account for kernel size
    point_mask = np.ones(np.array(amax.shape, np.int32) + kernel_size * 2, bool)

    iterate_kernel(score_argsorted, point_mask, kernel)

    amax_indices = amax_indices[
        point_mask[kernel_size:-kernel_size, kernel_size:-kernel_size]
    ]

    empty = np.zeros_like(img, bool)
    empty[amax_indices[..., 0], amax_indices[..., 1]] = True  # amax[point_mask]
    empty = empty[: score.shape[0], : score.shape[1]]

    # imu.imshow((img / (8 / 255)).astype(np.uint8))
    imu.imshow(empty.astype(np.uint8) * 255)

    return np.stack(np.where(empty), 1, dtype=np.int32)


def main(img, alpha):
    binary_mask = alpha > 0

    corner_val = calculate_point_score(img, alpha, binary_mask)
    points = narrow_down_points(corner_val, 16)
    # lower kernel size means less equal distance between points, kernel size of 2 works well, 1 works, but is noticably less equal
    points = narrow_down_points_cell_based(corner_val, 6, 2)

    # # visualize points
    # plot_img = img.copy()
    # for point in points:
    #     cv.circle(
    #         plot_img,
    #         point[::-1],
    #         2,
    #         (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
    #         cv.FILLED,
    #     )
    # imu.imshow(plot_img)

    # triangulate points using delaunay triangulation
    triangles = scp.Delaunay(points)
    indices = triangles.simplices

    # visualize triangulation
    plot_img = img.copy()
    for tri in indices:
        for i in range(tri.size):
            p0, p1 = points[tri[i]], points[tri[(i + 1) % tri.size]]
            cv.line(
                plot_img,
                p0[::-1],
                p1[::-1],
                (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
                1,
                cv.LINE_AA,
            )
    imu.imshow(plot_img)


if __name__ == "__main__":
    img = imu.imread("test_imgs/emote_upscaled.png", alpha=True)
    # img = cv.resize(img, (0, 0), fx=4, fy=4)

    alpha = img[..., -1]
    img = img[..., :3]

    for _ in range(1):
        main(img, alpha)
