def normalize_float_image(image):
    normalized_image = image
    normalized_image -= normalized_image.min()
    normalized_image /= normalized_image.max()
    return normalized_image
