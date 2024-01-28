import numpy as np
from multiprocessing import Pool, cpu_count


def compute_integral_image_chunk(chunk):
    integral_image, color_channel, chunk_indices = chunk
    for x, y in chunk_indices:
        integral_image[x][y] = sum([color_channel[i][j] for i in range(0, x + 1) for j in range(0, y + 1)])
    return integral_image


def partition_work(image_shape, num_chunks):
    chunk_size_x = image_shape[0] // num_chunks
    chunk_size_y = image_shape[1] // num_chunks
    chunks = []
    for i in range(num_chunks):
        for j in range(num_chunks):
            start_x = i * chunk_size_x
            end_x = start_x + chunk_size_x if i < num_chunks - 1 else image_shape[0]
            start_y = j * chunk_size_y
            end_y = start_y + chunk_size_y if j < num_chunks - 1 else image_shape[1]
            chunk_indices = [(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y)]
            chunks.append(chunk_indices)
    return chunks


def divide_and_conquer(image, num_chunks=None):
    if num_chunks is None:
        num_chunks = cpu_count()

    image_array = np.array(image)
    red = image_array[:, :, 0].astype(np.int64)
    green = image_array[:, :, 1].astype(np.int64)
    blue = image_array[:, :, 2].astype(np.int64)
    integral_images = []

    for color_channel in [red, green, blue]:
        integral_image = np.zeros_like(color_channel)
        chunks = partition_work(color_channel.shape, num_chunks)
        with Pool(processes=num_chunks) as pool:
            result_chunks = pool.map(compute_integral_image_chunk,
                                     [(integral_image, color_channel, chunk) for chunk in chunks])
        pool.close()
        pool.join()
        for chunk_result in result_chunks:
            integral_image += chunk_result
        print(integral_image)
        integral_images.append(integral_image)

    integral_img_r, integral_img_g, integral_img_b = integral_images
    return integral_img_r, integral_img_g, integral_img_b
