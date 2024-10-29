from PIL import Image
import multiprocessing as mp
import numpy as np
import time

def apply_sepia(img_array):

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = img_array.dot(sepia_matrix.T)
    np.putmask(sepia_img, sepia_img > 255, 255)  # Limit values to 255
    return sepia_img.astype(np.uint8)

def split_image(image, parts):
    img_array = np.array(image)
    height = img_array.shape[0]
    part_height = height // parts
    image_parts = []


    for i in range(parts):
        start_row = i * part_height
        if i < parts - 1:
            end_row = start_row + part_height
        else:
            end_row = height
        image_parts.append((i, img_array[start_row:end_row]))

    return image_parts

def process_image_part(args):
    idx, img_part = args
    return idx, apply_sepia(img_part)

def merge_parts(parts, shape):
    merged_image = np.zeros(shape, dtype=np.uint8)
    parts.sort(key=lambda x: x[0])
    part_height = shape[0] // len(parts)


    for i, part in parts:
        start_row = i * part_height
        merged_image[start_row:start_row + part.shape[0]] = part

    return Image.fromarray(merged_image)

def apply_sepia_parallel(image_path, output_path, num_processes=4):
    start_time = time.time()
    image = Image.open(image_path).convert('RGB')


    image_parts = split_image(image, num_processes)


    with mp.Pool(processes=num_processes) as pool:
        processed_parts = pool.map(process_image_part, image_parts)


    result_image = merge_parts(processed_parts, np.array(image).shape)
    result_image.save(output_path)

    print(f"Processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    input_path = "img3.png"
    output_path = "output_sepia.jpg"
    apply_sepia_parallel(input_path, output_path)
