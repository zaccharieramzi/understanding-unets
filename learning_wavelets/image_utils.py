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
