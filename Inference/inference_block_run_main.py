import os
import sys
import time
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
import psutil

# Import your custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.preprocessor import Preprocessor
from Model.model_utils import load_trained_model
from Inference.generate_samples_inference import generate_samples


# -------------------- CONFIGURATION --------------------
test_mode = True
test_batches = 8
test_data_path = "Training_Data/Synthetic_Loc_Data/test_data.json"
model_name = "Qwen/Qwen2.5-7B-Instruct"
main_model_dir = "saved_model_qwen7b_2e-06_32_0_20250529_230410"  # Main directory containing all epochs

# -------------------- DATA LOADING --------------------
custom_dataloader = CustomDataLoader(test_data_path)
data = custom_dataloader.load_all()
print(f"Loaded {len(data)} test samples")

# List of model configurations to run
model_configs = [
    {"name": "base_model", "path": None},  # Base model (no fine-tuning)
    {"name": "epoch1_full", "path": f"{main_model_dir}/epoch1_full"},
    {"name": "epoch2_full", "path": f"{main_model_dir}/epoch2_full"},
    {"name": "epoch3_full", "path": f"{main_model_dir}/epoch3_full"},
    {"name": "epoch4_full", "path": f"{main_model_dir}/epoch4_full"},
    {"name": "epoch5_full", "path": f"{main_model_dir}/epoch5_full"},
]

# -------------------- RUN INFERENCE FOR EACH MODEL --------------------
for config in model_configs:
    model_name_for_output = config["name"]
    trained_model_path = config["path"]
    
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name_for_output}")
    print(f"{'='*50}")
    
    # Set up output path
    if model_name_for_output == "base_model":
        inference_output = f"Inference_output/{main_model_dir}/base_model/generated_samples.json"
    else:
        inference_output = f"Inference_output/{main_model_dir}/{model_name_for_output}/generated_samples.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(inference_output), exist_ok=True)
    
    # Get the absolute path to your model directory if using fine-tuned model
    if trained_model_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)  # Move up one level if needed
        trained_model_path = os.path.join(project_dir, trained_model_path)
    
    # Load model and tokenizer
    print(f"Loading model from {'base model' if trained_model_path is None else trained_model_path}")
    model, tokenizer = load_trained_model(
        base_model_name=model_name,
        trained_model_path=trained_model_path,
        quantization_bits=0,
        device_map="auto"
    )
    
    print(f"Model loaded, torch dtype: {model.dtype}, device map: {model.hf_device_map}")
    
    # -------------------- GENERATION --------------------
    preprocessor = Preprocessor()
    
    print(f"Starting generation...")
    generated_samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        test_data=data,
        batch_size=4,
        max_length=1000,
        temperature=0.1,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        output_path=inference_output,
        test_mode=test_mode,
        test_batches=test_batches,
        num_beams=3
    )
    
    print(f"Generation finished. Results saved to {inference_output}")
    
    # Clean up to prevent CUDA OOM errors
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print memory usage
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Optional: Sleep for a moment to let system stabilize
    time.sleep(2)

print("\nAll inference runs completed!")