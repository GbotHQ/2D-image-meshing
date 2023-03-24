import numpy as np
import cv2 as cv
import imutil as imu
import itertools
from scipy import spatial as scp
import random as rng


seed = 42
rng.seed(seed)
np.random.seed(seed)


def calculate_point_score(img, alpha, binary_mask):
    # calculate corner score
    corner_val = cv.cornerMinEigenVal(
        imu.gray_mean(img.copy()).astype(np.uint8) * alpha, 3
    )
    corner_val -= np.amin(corner_val)
    corner_val /= np.amax(corner_val)
    corner_val += 1
    corner_val = np.where(
        binary_mask, corner_val, np.random.uniform(0, 1, (corner_val.shape))
    )

    # calculate offset to ensure points at image boundary
    offset = np.ones_like(corner_val)
    # prioritize points on opaque areas over transparent
    offset = np.where(binary_mask, offset, 0)

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


def narrow_down_points(corner_val, radius):
    radius = radius
    diameter = radius * 2
    kernel_size = diameter + 1

    # sort corner score in descending order
    argsorted_corner_val = np.unravel_index(
        np.argsort(corner_val, axis=None), corner_val.shape
    )
    argsorted_corner_val = np.stack(argsorted_corner_val, 1, dtype=np.int32)[::-1]
    # create a point mask with padding to account for kernel size
    point_mask = np.ones(
        (corner_val.shape[0] + diameter, corner_val.shape[1] + diameter), bool
    )

    for x, y in argsorted_corner_val:
        if not point_mask[x + radius, y + radius]:
            continue
        point_mask[x : x + kernel_size, y : y + kernel_size] = False
        point_mask[x + radius, y + radius] = True

    return np.stack(np.where(point_mask[radius:, radius:]), 1, dtype=np.int32)


def main(img, alpha, radius):
    binary_mask = alpha > 0

    corner_val = calculate_point_score(img, alpha, binary_mask)
    points = narrow_down_points(corner_val, radius)

    # # visualize features
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
    radius = 12

    img = imu.imread("test_imgs/emote_upscaled.png", alpha=True)
    # img = cv.resize(img, (0, 0), fx=1, fy=1)

    alpha = img[..., -1]
    img = img[..., :3]

    for _ in range(1):
        main(img, alpha, radius)
