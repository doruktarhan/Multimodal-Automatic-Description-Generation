import os
import sys
import time
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from torch.optim import AdamW
import gc
import psutil
import datetime

# Import your custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.dataset import RealEstateDataset, collate_fn
from Data.preprocessor import Preprocessor
from Model.model_utils import (
    load_model, load_tokenizer, create_lora_config, save_model, load_trained_model
)
from Evaluation.train_eval_metrics import evaluate_model
from Inference.generate_samples_inference import generate_samples


# -------------------- CONFIGURATION --------------------
# Set this to True to use the base model without fine-tuning
USE_BASE_MODEL = False  # Change this to True to use base model
test_mode = True
test_batches = 2

test_data_path = "Data/test_data.json"
model_name = "Qwen/Qwen2.5-3B-Instruct"
trained_model_short_path = "saved_model_qwen3b_1e-05_64_0_20250417_002746/epoch5_full"


# Model paths and settings
# Model paths and settings
if not USE_BASE_MODEL:
    # Path to fine-tuned model (relative to project root)
    inference_output = f"Inference_output/{trained_model_short_path}/generated_samples.json"
else:
    # Strip the epoch part from the path to get the base directory
    # Example: "saved_model_qwen3b_5e-05_32_0_20250416_140314/epoch3_full" -> "saved_model_qwen3b_5e-05_32_0_20250416_140314"
    base_dir = trained_model_short_path.split('/')[0]  # Get only the first part before the slash
    inference_output = f"Inference_output/{base_dir}/base_model/generated_samples.json"

# -------------------- DATA LOADING --------------------
custom_dataloader = CustomDataLoader(test_data_path)
data = custom_dataloader.load_all()
print(f"Loaded {len(data)} test samples")

# -------------------- MODEL LOADING --------------------
# Get the absolute path to your model directory if using fine-tuned model
if trained_model_short_path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)  # Move up one level if needed
    trained_model_path = os.path.join(project_dir, trained_model_short_path)
else:
    trained_model_path = None  # This will load the base model

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(inference_output), exist_ok=True)

# Load model and tokenizer
model, tokenizer = load_trained_model(
    base_model_name=model_name,
    trained_model_path=trained_model_path,
    quantization_bits=0,
    device_map="auto"
)

model_type = "base model" if USE_BASE_MODEL else f"fine-tuned model from {trained_model_short_path}"
print(f"Using {model_type}")
print(f"Model loaded from {model_name}, torch dtype: {model.dtype}, device map: {model.hf_device_map}")

# -------------------- GENERATION --------------------
preprocessor = Preprocessor()

generated_samples = generate_samples(
    model=model,
    tokenizer=tokenizer,
    preprocessor=preprocessor,
    test_data=data,
    batch_size=8,
    max_length=1000,
    temperature=0.1,  # Doesn't matter
    top_k=50,        # Doesn't matter
    top_p=0.9,        # Doesn't matter when do sample is false
    do_sample=True,   # For greedy decoding
    output_path=inference_output,
    test_mode = test_mode,
    test_batches = test_batches,
    num_beams = 3
)

print(f"Generation finished. Results saved to {inference_output}")