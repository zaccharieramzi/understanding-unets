import numpy as np


def normalize_float_image(image):
    normalized_image = np.copy(image)
    normalized_image -= normalized_image.min()
    normalized_image /= normalized_image.max()
    return normalized_image


def normalize_float_images(ref_image, *images):
    """Normalize images with a reference image"""
    normalized_images = list()
    normalized_image = np.copy(ref_image)
    im_min = normalized_image.min()
    normalized_image -= im_min
    im_max = normalized_image.max()
    normalized_image /= im_max
    normalized_images.append(normalized_image)
    for image in images:
        normalized_images.append(
            (image - im_min) / im_max
        )
    return normalized_images


def trim_zero_padding(*images, ref_index=0, zero_value=0):
    ref_image = images[ref_index]
    cols_to_remove = ~(np.squeeze(ref_image) == zero_value).all(axis=0)
    lines_to_remove = ~(np.squeeze(ref_image) == zero_value).all(axis=1)
    trimmed_images = [image[lines_to_remove][:, cols_to_remove] for image in images]
    return trimmed_images

def trim_padding(im_shape, *images):
    trimmed_images = []
    for image in images:
        padded_im_shape = image.shape[1:3]
        if (padded_im_shape != im_shape).any():
            to_trim = padded_im_shape - im_shape[0]
            trimmed_image = image[:, to_trim[0]//2:-to_trim[0]//2, to_trim[1]//2:-to_trim[1]//2]
        else:
            trimmed_image = image
        trimmed_images.append(trimmed_image)
    return trimmed_images
