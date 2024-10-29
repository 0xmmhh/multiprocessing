from PIL import Image
import multiprocessing as mp
import numpy as np
from typing import Tuple, List
import time

def apply_sepia(img_array: np.ndarray) -> np.ndarray:
    # Współczynniki do uzyskania efektu sepii
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Przekształcenie obrazu
    sepia_img = img_array.dot(sepia_matrix.T)
    
    # Przycinanie wartości do zakresu 0-255
    np.putmask(sepia_img, sepia_img > 255, 255)
    
    return sepia_img.astype(np.uint8)

def split_image(image: Image.Image, parts: int) -> List[Tuple[int, np.ndarray]]:
    img_array = np.array(image)
    height = img_array.shape[0]
    part_height = height // parts
    
    image_parts = []
    for i in range(parts):
        start = i * part_height
        end = start + part_height if i < parts - 1 else height
        image_parts.append((i, img_array[start:end]))
    
    return image_parts

def process_image_part(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    idx, img_part = args
    return idx, apply_sepia(img_part)

def merge_parts(parts: List[Tuple[int, np.ndarray]], shape: Tuple) -> Image.Image:
    # Tworzymy pustą tablicę o wymiarach oryginalnego obrazu
    merged = np.zeros(shape, dtype=np.uint8)
    
    # Sortujemy części według indeksu
    parts.sort(key=lambda x: x[0])
    
    # Obliczamy wysokość jednej części
    part_height = shape[0] // len(parts)
    
    # Wstawiamy każdą część na odpowiednie miejsce
    for i, part in parts:
        start = i * part_height
        end = start + part.shape[0]
        merged[start:end] = part
    
    return Image.fromarray(merged)

def apply_sepia_parallel(image_path: str, output_path: str, num_processes: int = 4):
    # Wczytanie obrazu
    start_time = time.time()
    image = Image.open(image_path).convert('RGB')
    
    # Podział obrazu na części
    image_parts = split_image(image, num_processes)
    
    # Utworzenie puli procesów
    with mp.Pool(processes=num_processes) as pool:
        # Przetwarzanie części obrazu równolegle
        processed_parts = pool.map(process_image_part, image_parts)
    
    # Połączenie przetworzonych części
    result_image = merge_parts(processed_parts, np.array(image).shape)
    
    # Zapisanie wyniku
    result_image.save(output_path)
    
    end_time = time.time()
    print(f"Czas przetwarzania: {end_time - start_time:.2f} sekund")

if __name__ == '__main__':
    # Przykład użycia
    input_path = "3097725.jpg"
    output_path = "output_sepia.jpg"
    apply_sepia_parallel(input_path, output_path)