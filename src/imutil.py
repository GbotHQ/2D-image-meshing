from multiprocessing.sharedctypes import Value
import os
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


def to_float(img, dtype=np.float32):
    """convert uint to float"""
    return img.astype(dtype) / np.iinfo(img.dtype).max


def to_uint(img, dtype=np.uint8):
    """convert float to uint"""
    return (img * np.iinfo(dtype).max).astype(dtype)


def srgb_to_lin(s):
    """convert srgb to linear"""
    lin0 = s / 12.92
    lin1 = ((s + 0.055) / 1.055) ** 2.4
    lin = np.where(s <= 0.0404482362771082, lin0, lin1)
    return lin


def lin_to_srgb(lin):
    """convert linear to srgb"""
    s0 = 1.055 * (lin ** (1 / 2.4)) - 0.055
    s1 = 12.92 * lin
    s = np.where(lin > 0.0031308, s0, s1)
    return s


def __convert_for_write(img, linear):
    is_float = np.issubdtype(img.dtype, np.floating)

    if linear and not is_float:
        img = to_float(img)
        is_float = True

    if linear:
        img = lin_to_srgb(img)

    if is_float:
        img = to_uint(saturate(img))

    return img


def imshow(img, linear=False):
    img = __convert_for_write(img, linear)

    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def imshow_grid(imgs, downscale=1, linear=False):
    if downscale > 1:
        imgs = [downsample(img=e, downscale=downscale) for e in imgs]
    imshow(np.concatenate(imgs, axis=1), linear)


def downsample(img, downscale):
    return cv.resize(img, np.array(img.shape[:2], np.int32) // downscale, interpolation=cv.INTER_AREA)


def __imwrite_check(imwrite_result, path):
    if not imwrite_result:
        raise Exception(f"Can't write image to {path}")


def imwrite(path, img, linear=False, **kwargs):
    img = __convert_for_write(img, linear)

    __imwrite_check(cv.imwrite(path, img, **kwargs), path)


def mix(a, b, fac):
    fac = ensure_dims(fac, [a, b])
    return a * (1 - fac) + b * fac


def smoothstep(x, edge0, edge1):
    x1 = (x.copy() - edge0) / (edge1 - edge0)
    x1 = x1 * x1 * (3 - 2 * x1)

    return np.where(x >= edge1, 1.0, np.where(x < edge0, 0.0, x1))


def normalize_img(arr):
    min = np.amin(arr)
    max = np.amax(arr)

    return (arr - min) / (max - min)


def __imread_check(img, path):
    if img is None:
        raise Exception(f"Can't read {path} image")


def imread(path, fdtype=None, alpha=False):
    img = cv.imread(path, cv.IMREAD_UNCHANGED if alpha else cv.IMREAD_COLOR)
    __imread_check(img, path)

    if fdtype is None:
        return img
    elif np.issubdtype(fdtype, np.floating):
        return to_float(img, fdtype)
    else:
        raise Exception(f"dtype {fdtype} must be float.")


def imread_exr(path, alpha=False):
    """read an exr image"""
    flags = cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH
    if alpha:
        flags = flags | cv.IMREAD_UNCHANGED
    img = cv.imread(str(path), flags)
    __imread_check(img, str(path))
    return img


def swap_channels(img, *channels):
    return np.dstack([img[..., c] for c in channels])


def rgb_to_bgr(img):
    if img.shape[-1] == 3:
        return swap_channels(img, 2, 1, 0)
    elif img.shape[-1] == 4:
        return swap_channels(img, 2, 1, 0, 3)
    else:
        raise Exception(
            f"Input image must have 3 or 4 channels but got {img.shape[-1]}"
        )


def bgr_to_rgb(img):
    return rgb_to_bgr(img)


def normalize(vector_array):
    return safe_divide(
        vector_array, np.linalg.norm(vector_array, axis=-1, keepdims=True), thresh=0
    )


def ensure_dims(src, ref):
    """Ensure that src image has the same number of dimensions as ref image"""
    reference = None

    if type(ref) is list:
        for v in ref:
            if isinstance(v, np.ndarray):
                reference = v
                break

        if reference is None:
            raise ValueError("At least one of the inputs must be a numpy array")
    else:
        reference = ref

    ref_dims = len(reference.shape)
    src_dims = len(src.shape)

    if ref_dims == src_dims:
        return src
    elif ref_dims == src_dims + 1:
        return src[..., None]
    else:
        raise ValueError("ensure_dims() only works on 2d images")


def gray_mean(img, dtype=np.float32):
    """Convert to grayscale by mean of RGB"""
    return np.mean(img, axis=2, dtype=dtype)


def saturate(img):
    """Clamp image in 0 to 1 range"""
    return np.clip(img, 0, 1)


def sort_pixels_bright(imgs, dtype=np.int8):
    multimask = np.argsort(imgs, axis=0)
    return multimask.astype(dtype)


def multimask_to_mask_set(multimask, n_masks):
    """Create individual mask from a multimask"""
    masks = []
    for i in range(n_masks):
        mask = 1 - np.absolute(multimask - i)  # linear falloff
        mask = np.maximum(mask, 0)  # clamp

        masks.append(mask)

    return masks


def apply_index_map(arrs, indices):
    masks = multimask_to_mask_set(indices, len(arrs))

    result = np.zeros_like(arrs[0])
    for array, mask in zip(arrs, masks):
        mask = ensure_dims(mask, array)
        result += array * mask

    return result


def safe_divide(num, denom, thresh=0):
    """Divide without NaNs on division by 0"""
    return np.divide(num, denom, where=denom > thresh)


def alpha_pad(img, alpha):
    """Binary alpha padding"""
    inpainted = np.zeros_like(img)

    for i in range(inpainted.shape[2]):
        inpainted[..., i] = cv.inpaint(
            img[..., i], (alpha < 1).astype(np.uint8), 1, cv.INPAINT_NS
        )
    return inpainted
