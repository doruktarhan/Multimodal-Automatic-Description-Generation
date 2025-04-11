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


test_data_path = "Data/dummy/dummy_test_data.json"


custom_dataloader = CustomDataLoader(test_data_path)

data = custom_dataloader.load_all()
print(len(data))

model_name = "Qwen/Qwen2.5-3B-Instruct"
trained_data_path = None

#load the trained model if train data path exists
model, tokenizer = load_trained_model(
    base_model_name=model_name,
    trained_model_path=trained_data_path,
    quantization_bits=0,
    device_map="cpu"
)

print(f"Model loaded from {model_name}, torch dtype: {model.dtype}, device map: {model.hf_device_map}")



preprocessor = Preprocessor()


generated_samples = generate_samples(
    model=model,
    tokenizer= tokenizer,
    preprocessor=preprocessor,
    test_data=data,
    batch_size=1,
    max_length=200,
    temperature=None,  # Doesn't matter
    top_k=None,         # Doesn't matter
    top_p=None,         # Doesn't matter when do sample is false
    do_sample=False, #for greedy decoding
)

