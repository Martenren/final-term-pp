import numpy as np


def integral_image_vectorized(image):
    # Convert the image to numpy array
    image_array = np.array(image)

    # Compute the integral image for each color channel
    red_integral = np.cumsum(np.cumsum(image_array[:, :, 0], axis=0), axis=1)
    green_integral = np.cumsum(np.cumsum(image_array[:, :, 1], axis=0), axis=1)
    blue_integral = np.cumsum(np.cumsum(image_array[:, :, 2], axis=0), axis=1)

    return red_integral, green_integral, blue_integral

