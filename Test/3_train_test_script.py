###########################
# train.py (example)
###########################

import os
import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import wandb  # We'll show how to integrate W&B
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import your custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.dataset import RealEstateDataset, collate_fn
from Data.preprocessor import Preprocessor
from Model.model_utils import (
    load_model, load_tokenizer, create_lora_config, save_model
)
from Utils.memory_tracking_utils import print_memory_usage,cleanup_memory




def main():
    ########################################
    # 1. Set hyperparameters + config
    ########################################
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    
    # Training hyperparams (feel free to tweak)
    stream_chunk_size = 5   # Number of samples to read from JSON each time
    micro_batch_size = 1      # Actual batch size per step
    epochs_per_chunk = 1      # How many epochs to run on each chunk
    learning_rate = 1e-4
    total_chunks_to_train = None  # e.g., None means train on all chunks

    output_dir = "saved_lora_model"
    
    # (Optional) Lora config
    lora_cfg = create_lora_config(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    ########################################
    # 2. Initialize experiment tracking (W&B)
    ########################################
    # If you haven't used wandb before:
    # 1) `pip install wandb`
    # 2) `wandb login` at command line
    # Then you can do:
    
    # wandb.init(
    #     project="RealEstate-LLM-Finetuning",
    #     name="Qwen-finetuning-test",  # A run name
    #     config={
    #         "model_name": model_name,
    #         "learning_rate": learning_rate,
    #         "batch_size": micro_batch_size,
    #         "stream_chunk_size": stream_chunk_size,
    #     }
    # )
    # wandb is now tracking your runs; you can see logs on your W&B dashboard

    ########################################
    # 3. Load model + tokenizer
    ########################################
    tokenizer = load_tokenizer(model_name)
    # e.g., for 4-bit quant + LoRA
    model = load_model(
        model_name=model_name,
        use_lora=False,
        lora_config=lora_cfg,
        quantization_bits=0,   # or 8, or 0 for no quant
        device_map="cpu"      # auto device placement
    )

    # Let’s assume the model is on GPU automatically if you have one;
    # otherwise specify model.to(device) if needed.

    ########################################
    # 4. Create optimizer + (optional) scheduler
    ########################################
    # Normally you only need to optimize the LoRA layers if using LoRA
    # But if you want the entire model to be trainable, you'd do model.parameters()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # If you know total steps, you can create a scheduler. For streaming,
    # you might not know total steps in advance. Let's assume we do something basic:
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=100,  # small warmup
    #     num_training_steps=??? # you must decide a rough total
    # )

    ########################################
    # 5. Stream data + training loop
    ########################################
    preprocessor = Preprocessor()
    data_loader = CustomDataLoader(data_path)
    
    chunk_count = 0

    # We'll iterate over each chunk from the custom loader
    for raw_chunk in data_loader.stream_data(batch_size=stream_chunk_size):
        chunk_count += 1
        if total_chunks_to_train is not None and chunk_count > total_chunks_to_train:
            break  # stop early if user-limited

        # Preprocess chunk
        processed_examples = preprocessor.process_data(raw_chunk)

        # Create the dataset
        dataset = RealEstateDataset.from_preprocessor(
            raw_data=raw_chunk,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            max_input_length=512,
            max_output_length=512
        )

        # Wrap in a PyTorch DataLoader for mini-batching
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Train for X epochs on this chunk
        for epoch in range(epochs_per_chunk):
            for step, batch in enumerate(train_loader):
                # Move data to GPU (if needed)
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # if scheduler: scheduler.step()

                # Print or log info
                if step % 10 == 0:
                    print(f"[Chunk {chunk_count} | Epoch {epoch} | Step {step}] loss = {loss.item():.4f}")
                    #wandb.log({"loss": loss.item()})  # logs to W&B

        # (Optional) Save partial checkpoint after finishing this chunk
        partial_dir = os.path.join(output_dir, f"checkpoint_chunk_{chunk_count}")
        save_model(model, partial_dir)

    ########################################
    # 6. Final model save + cleanup
    ########################################
    save_model(model, output_dir)
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
