import os
from PIL import Image
import numpy as np
import time
from multiprocessing import Pool, Process, Queue
import matplotlib.pyplot as plt
from matplotlib import colormaps


def compute_columns(args, result_queue):
    integral_image, color_channel = args
    for y in range(1, color_channel.shape[1]):
        integral_image[0][y] = color_channel[0][y] + integral_image[0][y - 1]

    result_queue.put(integral_image)


def compute_rows(args, result_queue):
    integral_image, color_channel = args
    for x in range(1, color_channel.shape[0]):
        integral_image[x][0] = color_channel[x][0] + integral_image[x - 1][0]

    result_queue.put(integral_image)


def compute_integral_image(integral_image, color_channel, x, y):
    integral_image[x][y] = (color_channel[x][y] + integral_image[x - 1][y] + integral_image[x][y - 1] -
                            integral_image[x - 1][y - 1])
    return integral_image, x, y


def compute_integral_image_parallel(color_channel):
    integral_image = [[0 for x in range(color_channel.shape[1])] for y in range(color_channel.shape[0])]
    integral_image[0][0] = color_channel[0][0]

    result_queue = Queue()

    p1 = Process(target=compute_columns, args=((integral_image, color_channel), result_queue))
    p2 = Process(target=compute_rows, args=((integral_image, color_channel), result_queue))

    p1.start()
    p2.start()

    result1 = result_queue.get()
    result2 = result_queue.get()

    print("horizontal, vertical done")

    integral_image = np.subtract(np.add(result1, result2), integral_image)

    p1.join()
    p2.join()

    rows = len(integral_image)

    diagonals = []
    for x in range(1, rows - 1):
        diagonal_indices = np.column_stack((np.arange(x, 0, -1), np.arange(1, x + 1)))
        diagonals.append(diagonal_indices.tolist())

    for diagonal in diagonals:
        with Pool(processes=os.cpu_count()) as pool:
            subsets_integral_images = pool.starmap(compute_integral_image,
                                                   [(integral_image, color_channel, x, y) for x, y in diagonal])

            for j in range(0, len(subsets_integral_images)):
                x_, y_ = subsets_integral_images[j][1], subsets_integral_images[j][2]
                integral_image[x_, y_] = subsets_integral_images[j][0][x_][y_]
            pool.close()
            pool.join()
        print("diagonal:", diagonals.index(diagonal) + 1, "done", len(diagonals) - (diagonals.index(diagonal) + 1),
              "left")

    print("channel done")
    return integral_image


def parallel_integral_image_color(image):
    image_array = np.array(image)

    red = image_array[:, :, 0].astype(np.int64)
    green = image_array[:, :, 1].astype(np.int64)
    blue = image_array[:, :, 2].astype(np.int64)

    result = [compute_integral_image_parallel(color_channel) for color_channel in [red, green, blue]]

    integral_img_r, integral_img_g, integral_img_b = result

    return integral_img_r, integral_img_g, integral_img_b


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


if __name__ == '__main__':
    input_image_path = "img/5600-3200.jpg"
    image_name = input_image_path.split(".")
    image_name = image_name[0].split("/")[1]
    output_image_path_r = f"img/{image_name}_red.jpg"
    output_image_path_g = f"img/{image_name}_green.jpg"
    output_image_path_b = f"img/{image_name}_blue.jpg"
    color_image = Image.open(input_image_path)

    image_array = []
    color_map = colormaps['Set2']

    option = input("Parallel (y/n): ")
    if option == "y":
        p_type = input("Large image set or fast one image or different sizes: (l/f/d): ")
        save = input("Save images (y/n): ")
        if p_type == "l":
            max_cpu = os.cpu_count()
            for _ in range(1, max_cpu + 1):
                image_array.append(color_image)

            for i in range(3, 20):
                image_array = []
                for _ in range(1, i):
                    image_array.append(color_image)

                times = {}

                for cpu_count in range(1, max_cpu + 1):
                    start = time.time()
                    p = Pool(processes=cpu_count)
                    p.map(integral_image_color, [image for image in image_array])
                    end = time.time()
                    print(f"Time taken for {cpu_count} cores: {end - start}")

                    times[cpu_count] = end - start
                    p.close()

                if save == "y":
                    plt.xticks(list(times.keys()))
                    plt.bar(list(times.keys()), list(times.values()), color=color_map.colors)
                    plt.xlabel("Number of cores")
                    plt.ylabel("Time taken (s)")
                    plt.savefig(f'{len(image_array)}_images.png')
        elif p_type == "f":
            start = time.time()
            integral_image = parallel_integral_image_color(color_image)
            end = time.time()
            print(f"Time taken for parallel execution: {end - start}")

    else:
        start = time.time()
        integral_image_color(color_image)
        end = time.time()
        print(f"Time taken for sequential execution: {end - start}")
