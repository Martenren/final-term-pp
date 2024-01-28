import numpy as np


def integral_image_color(image):
    image_array = np.array(image)

    def calculate_integral_image(img_array):
        integral_img = [[0 for x in range(red.shape[1])] for y in range(red.shape[0])]

        for x in range(0, img_array.shape[0]):
            for y in range(0, img_array.shape[1]):
                if x == 0 and y == 0:
                    integral_img[x][y] = img_array[x][y]
                elif x == 0:
                    integral_img[x][y] = img_array[x][y] + integral_img[x][y - 1]
                elif y == 0:
                    integral_img[x][y] = img_array[x][y] + integral_img[x - 1][y]
                else:
                    integral_img[x][y] = (img_array[x][y] + integral_img[x - 1][y] + integral_img[x][y - 1] -
                                          integral_img[x - 1][y - 1])
        return np.array(integral_img)

    # Initialize arrays for each color channel
    red = image_array[:, :, 0].astype(np.int64)
    green = image_array[:, :, 1].astype(np.int64)
    blue = image_array[:, :, 2].astype(np.int64)

    result = [calculate_integral_image(color_channel) for color_channel in [red, green, blue]]

    integral_img_r, integral_img_g, integral_img_b = result

    return integral_img_r, integral_img_g, integral_img_b