import gc
import sys
import torch
import psutil
import os
import shutil

import subprocess

def show_huggingface_cache_sizes():
    """
    Print the size of each subfolder under ~/.cache/huggingface
    and then print the total size of the entire huggingface cache.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if not os.path.exists(cache_dir):
        print(f"No huggingface cache directory found at {cache_dir}")
        return
    
    print(f"Showing per-subfolder usage in: {cache_dir}\n")
    
    try:
        # Show each subfolder’s size (datasets, hub, etc.)
        subfolders_output = subprocess.check_output(["du", "-sh", f"{cache_dir}/*"])
        print(subfolders_output.decode("utf-8"))
        
        # Show total usage of the entire huggingface cache
        total_output = subprocess.check_output(["du", "-sh", cache_dir])
        total_size = total_output.decode("utf-8").strip()
        print(f"Total huggingface cache usage: {total_size}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error calling 'du' command: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def print_memory_usage(step_name):
    """Print detailed memory usage statistics"""
    # Python memory
    process = psutil.Process(os.getpid())
    python_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    # CPU memory
    cpu_percent = psutil.virtual_memory().percent
    cpu_used = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
    cpu_total = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    
    print(f"\n----- Memory Usage at {step_name} -----")
    print(f"Python process memory: {python_mem:.2f} MB")
    print(f"CPU memory: {cpu_used:.2f}/{cpu_total:.2f} GB ({cpu_percent}%)")
    
    # Count PyTorch tensors
    total_size = 0
    tensor_count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor_count += 1
                total_size += obj.element_size() * obj.nelement()
        except:
            pass
    
    print(f"Number of PyTorch tensors: {tensor_count}")
    print(f"Total tensor memory: {total_size / (1024 * 1024):.2f} MB")
    print("-------------------------------\n")
    
    # Collect garbage to potentially free memory
    gc.collect()

def cleanup_memory():
    """Aggressively clean up memory"""
    gc.collect()
    
    # Force deallocate unused tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                # Check if tensor isn't referenced by anything else
                if sys.getrefcount(obj) <= 2:
                    obj.detach_()
        except:
            pass
    gc.collect()


def clear_huggingface_cache():
    # The default cache directory for Hugging Face models
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if os.path.exists(cache_dir):
        print(f"Removing Hugging Face cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    else:
        print(f"Cache directory not found: {cache_dir}")


# Example usage in train.py:
# print_memory_usage("Before model loading")
# model = load_model(...)
# print_memory_usage("After model loading")
# 
# for epoch in range(epochs):
#     print_memory_usage(f"Start of epoch {epoch}")
#     for step, batch in enumerate(train_loader):
#         print_memory_usage(f"Start of batch {step}")
#         # ... training code ...
#         print_memory_usage(f"End of batch {step}")
#         cleanup_memory()