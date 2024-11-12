import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = 1000000000000

def apply_sepia(img_array):
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = img_array.dot(sepia_matrix.T)
    np.putmask(sepia_img, sepia_img > 255, 255)
    return sepia_img.astype(np.uint8)


def split_image(image, parts):
    img_array = np.array(image)
    height = img_array.shape[0]
    part_height = height // parts
    image_parts = []

    for i in range(parts):
        start_row = i * part_height
        end_row = start_row + part_height if i < parts - 1 else height
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
    # start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        processed_parts = pool.map(process_image_part, image_parts)
    # return time.time() - start_time
    result_image = merge_parts(processed_parts, np.array(image).shape)
    result_image.save(output_path)
    return time.time() - start_time

def apply_sepia_parallel_barebone(image_path, output_path, num_processes=4):

    image = Image.open(image_path).convert('RGB')
    image_parts = split_image(image, num_processes)
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        processed_parts = pool.map(process_image_part, image_parts)
    return time.time() - start_time



def apply_sepia_single(image_path, output_path):
    start_time = time.time()
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)

    sepia_image = apply_sepia(img_array)
    result_image = Image.fromarray(sepia_image)
    result_image.save(output_path)
    return time.time() - start_time


class SepiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sepia Image Converter")
        self.root.geometry("800x500")
        self.root.configure(bg="white")
        self.image_path = None

        self.preview_frame = tk.Frame(root, bg="white")
        self.preview_frame.place(relx=0.5, rely=0.4, anchor="center")

        self.original_label = tk.Label(self.preview_frame, text="Original Image", font=("Arial", 14), bg="white")
        self.original_label.grid(row=0, column=0, padx=10, pady=5)
        self.original_image_label = tk.Label(self.preview_frame, width=375, height=300, bg="lightgrey", relief="solid")
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)

        self.processed_label = tk.Label(self.preview_frame, text="Processed Image", font=("Arial", 14), bg="white")
        self.processed_label.grid(row=0, column=1, padx=10, pady=5)
        self.processed_image_label = tk.Label(self.preview_frame, width=375, height=300, bg="lightgrey", relief="solid")
        self.processed_image_label.grid(row=1, column=1, padx=10, pady=10)

        self.button_frame = tk.Frame(root, bg="white")
        self.button_frame.place(relx=0.5, rely=0.85, anchor="center")

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", font=("Arial", 12),
                                       command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=10)

        self.parallel_button = tk.Button(self.button_frame, text="Apply Sepia (Parallel)", font=("Arial", 12),
                                         command=self.apply_sepia_parallel)
        self.parallel_button.grid(row=0, column=1, padx=10)

        self.plot_button = tk.Button(self.button_frame, text="Create plot", font=("Arial", 12),
                                     command=self.plot_performance)
        self.plot_button.grid(row=0, column=2, padx=10)

        self.plot_button = tk.Button(self.button_frame, text="Create plot (barebone)", font=("Arial", 12),
                                     command=self.plot_performance)
        self.plot_button.grid(row=0, column=3, padx=10)

        self.single_button = tk.Button(self.button_frame, text="Apply Sepia (Single)", font=("Arial", 12),
                                       command=self.apply_sepia_single)
        self.single_button.grid(row=0, column=4, padx=10)

        self.time_label = tk.Label(root, text="", font=("Arial", 12), bg="white")
        self.time_label.place(relx=0.5, rely=0.95, anchor="center")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.cmyk")])
        if self.image_path:
            img = Image.open(self.image_path)
            img_resized = self.resize_image(img, 375, 300)
            self.original_image = ImageTk.PhotoImage(img_resized)
            self.original_image_label.config(image=self.original_image)
            self.processed_image_label.config(image='')
            self.time_label.config(text="")

    def resize_image(self, img, max_width, max_height):

        img_ratio = img.width / img.height
        label_ratio = max_width / max_height

        if img_ratio > label_ratio:
            new_width = max_width
            new_height = int(max_width / img_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * img_ratio)

        return img.resize((new_width, new_height), Image.LANCZOS)

    def apply_sepia_parallel(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        output_path = "output_sepia_parallel.jpg"
        processing_time = apply_sepia_parallel(self.image_path, output_path)
        self.show_processed_image(output_path, processing_time)

    def apply_sepia_single(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        output_path = "output_sepia_single.jpg"
        processing_time = apply_sepia_single(self.image_path, output_path)
        self.show_processed_image(output_path, processing_time)

    def show_processed_image(self, output_path, processing_time):
        img = Image.open(output_path)
        img_resized = self.resize_image(img, 375, 300)
        self.processed_image = ImageTk.PhotoImage(img_resized)
        self.processed_image_label.config(image=self.processed_image)
        self.time_label.config(text=f"Processing Time: {processing_time:.2f} seconds")

    def plot_performance(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        process_counts = [1, 2, 4, 6, 8, 16]
        execution_times = []
        for counts in process_counts:
            if counts == 1:
                execution_times.append(apply_sepia_single(self.image_path, 'output_sepia_single.jpg'))
            elif counts > 1:
                execution_times.append(apply_sepia_parallel(self.image_path, 'output_sepia_parallel.jpg', counts))
        plt.figure(figsize=(10, 5))
        plt.plot(process_counts, execution_times, marker='o', linestyle='-', color='b')
        plt.title("Execution Time vs Number of Processes")
        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(process_counts)
        plt.grid()
        plt.show()

    def plot_performance_barebone(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        process_counts = [1, 2, 4, 6, 8, 16]
        execution_times = []
        for counts in process_counts:
            if counts == 1:
                execution_times.append(apply_sepia_single(self.image_path, 'output_sepia_single.jpg'))
            elif counts > 1:
                execution_times.append(apply_sepia_parallel(self.image_path, 'output_sepia_parallel.jpg', counts))
        plt.figure(figsize=(10, 5))
        plt.plot(process_counts, execution_times, marker='o', linestyle='-', color='b')
        plt.title("Execution Time vs Number of Processes")
        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(process_counts)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = SepiaApp(root)
    root.mainloop()
    print(apply_sepia_parallel('image100.jpg', 'output_sepia_parallel.jpg', 4))
    print(apply_sepia_single('image100.jpg', 'output_sepia_single.jpg'))
