# benchmark_sepia.py
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
from file1 import apply_sepia_single, apply_sepia_parallel  # Import from your original file

def benchmark_sepia_processing(image_path, process_counts=[1, 2, 4, 6, 8, 12, 16]):
    """
    Benchmark sepia processing with different numbers of processes
    """
    times = []
    
    print("Starting benchmark...")
    print(f"Testing with {len(process_counts)} different process counts: {process_counts}")
    
    # Test each process count
    for num_processes in process_counts:
        if num_processes == 1:
            print(f"\nTesting with single process...")
            time_taken = apply_sepia_single(image_path, "output_single.jpg")
        else:
            print(f"\nTesting with {num_processes} processes...")
            time_taken = apply_sepia_parallel(image_path, f"output_parallel_{num_processes}.jpg", num_processes)
        
        times.append(time_taken)
        print(f"Time taken: {time_taken:.2f} seconds")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(process_counts, times, 'bo-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.title('Sepia Processing Time vs Number of Processes', fontsize=12, pad=15)
    plt.xlabel('Number of Processes', fontsize=10)
    plt.ylabel('Processing Time (seconds)', fontsize=10)
    
    # Add point annotations
    for i, (x, y) in enumerate(zip(process_counts, times)):
        plt.annotate(f'{y:.2f}s', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('sepia_benchmark.png')
    plt.close()
    
    return process_counts, times

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an image path")
        print("Usage: python benchmark_sepia.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    processes, timings = benchmark_sepia_processing(image_path)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("--------------------------")
    for proc, time in zip(processes, timings):
        print(f"{proc} process(es): {time:.2f} seconds")
    
    print(f"\nBenchmark plot saved as 'sepia_benchmark.png'")