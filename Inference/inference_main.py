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


test_data_path = "Data/test_data.json"

custom_dataloader = CustomDataLoader(test_data_path)

data = custom_dataloader.load_all()
print(len(data))

model_name = "Qwen/Qwen2.5-3B-Instruct"


import os

# Get the absolute path to your model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)  # Move up one level if needed
short_path = "saved_model_qwen3b_5e-05_64_8_20250411_183252/epoch1_full"
trained_model_path = os.path.join(project_dir, short_path)


# 8 bit error on quantization
inference_output = "Inference_output/" + short_path + "/generated_samples.json"
#load the trained model if train data path exists
model, tokenizer = load_trained_model(
    base_model_name=model_name,
    trained_model_path=trained_model_path,
    quantization_bits=0,
    device_map="auto"
)

print(f"Model loaded from {model_name}, torch dtype: {model.dtype}, device map: {model.hf_device_map}")

preprocessor = Preprocessor()

generated_samples = generate_samples(
    model=model,
    tokenizer= tokenizer,
    preprocessor=preprocessor,
    test_data=data,
    batch_size=8,
    max_length=1000,
    temperature=None,  # Doesn't matter
    top_k=None,         # Doesn't matter
    top_p=None,         # Doesn't matter when do sample is false
    do_sample=False, #for greedy decoding
    output_path = inference_output
)
print(f"Generation finished")
