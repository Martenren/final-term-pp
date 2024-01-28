import os
from PIL import Image
import numpy as np
import time
from multiprocessing import Pool, Process, Queue
import matplotlib.pyplot as plt
from matplotlib import colormaps

from parallel import parallel_integral_image_color
from divide_and_conquer import divide_and_conquer
from sequential import integral_image_color
from vectorization import integral_image_vectorized

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
        p_type = input("Large image set, fast one image v1, fast one image v2, vectorization: (l/f/f2/v): ")
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
        elif p_type == "f2":
            start = time.time()
            integral_image = divide_and_conquer(color_image)
            end = time.time()
            print(f"Time taken for parallel execution: {end - start}")
        elif p_type == "v" and save == "n":
            start = time.time()
            integral_image = integral_image_vectorized(color_image)
            end = time.time()
            print(f"Time taken for vectorized execution: {end - start}")

        elif p_type == "v" and save == "y":
            times_v = {}
            times_s = {}
            image_paths = [('img/500-500', '500-500'), ('img/1000-1000', '1000-1000'),
                           ('img/2000-2000', '2000-2000'), ('img/3000-3000', '3000-3000'),
                           ('img/5600-3200', '5600-3200')]

            for path, name in image_paths:
                color_image = Image.open(f"{path}.jpg")
                start = time.time()
                integral_image_v = integral_image_vectorized(color_image)
                end = time.time()
                print(f"time taken for vectorized: {end - start}, {name}")
                times_v[name] = end - start
                start_time = time.time()
                integral_image_c = integral_image_color(color_image)
                end_time = time.time()
                print(f"time taken for sequential: {end_time - start_time}, {name}")
                times_s[name] = end_time - start_time
                print(f"{name} increase in speed of {round((times_s[name]) / times_v[name], 2) * 100}%")
                print(f"{name} done")

            plt.plot(list(times_v.keys()), list(times_v.values()), color=color_map.colors[0], label="Vectorized")
            plt.plot(list(times_s.keys()), list(times_s.values()), color=color_map.colors[1], label="Sequential")
            plt.xlabel("Image size")
            plt.ylabel("Time taken (s)")
            plt.legend()
            plt.savefig(f'calculation_parallelisation/vect_vs_sequential.png')
            plt.show()

        elif "test":
            image_paths = [('img/500-500', '500-500'), ('img/1000-1000', '1000-1000'),
                           ('img/2000-2000', '2000-2000'), ('img/3000-3000', '3000-3000'),
                           ('img/5600-3200', '5600-3200')]
            times_v = {}
            times_p = {}
            times_pv = {}
            image_count = 8
            for path, name in image_paths:
                image = Image.open(f"{path}.jpg")
                color_images = [image for _ in range(1, image_count + 1)]

                start = time.time()
                integral_image_v = [integral_image_vectorized(image) for image in color_images]
                end = time.time()
                times_v[name] = end - start
                print(f"time taken for vectorized: {end - start}, {name}")
                start_time = time.time()
                with Pool(processes=os.cpu_count()) as pool:
                    pool.map(integral_image_color, [image for image in color_images])
                end_time = time.time()
                times_p[name] = end_time - start_time
                print(f"time taken for parallel: {end_time - start_time}, {name}")
                start_time = time.time()
                with Pool(processes=os.cpu_count()) as pool:
                    pool.map(integral_image_vectorized, [image for image in color_images])
                end_time = time.time()
                times_pv[name] = end_time - start_time
                print(f"time taken for parallel vectorized: {end_time - start_time}, {name}")
                print(f"{name} increase in speed vectorized vs parallel {round(times_p[name] / times_v[name], 2) * 100}%")
                print(f"{name} increase in speed vectorized vs parallel vectorized {round(times_pv[name] / times_v[name], 2) * 100}%")
                print(f"{name} done")

            plt.plot(list(times_v.keys()), list(times_v.values()), color=color_map.colors[0], label="Vectorized")
            plt.plot(list(times_p.keys()), list(times_p.values()), color=color_map.colors[1], label="Parallel")
            plt.plot(list(times_pv.keys()), list(times_pv.values()), color=color_map.colors[2], label="Parallel Vectorized")
            plt.xlabel("Image size")
            plt.ylabel("Time taken (s)")
            plt.legend()
            plt.savefig(f'calculation_parallelisation/vect_vs_batch_parallel.png')
            plt.title(f"batch {image_count} images: Vectorized vs Batch Parallel ({os.cpu_count()} cores)")
            plt.show()

    else:
        start = time.time()
        integral_image_color(color_image)
        end = time.time()
        print(f"Time taken for sequential execution: {end - start}")
