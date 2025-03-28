import os
import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from torch.optim import AdamW
import gc
import psutil

# Import your custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.dataset import RealEstateDataset, collate_fn
from Data.preprocessor import Preprocessor
from Model.model_utils import (
    load_model, load_tokenizer, create_lora_config, save_model
)

def print_memory_usage(label):
    """Print current memory usage with a descriptive label"""
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)  # MB
    
    # System memory
    vm = psutil.virtual_memory()
    sys_used = vm.used / (1024 * 1024 * 1024)  # GB
    sys_total = vm.total / (1024 * 1024 * 1024)  # GB
    
    print(f"\n----- MEMORY [{label}] -----")
    print(f"Process Memory: {rss:.2f} MB")
    print(f"System Memory: {sys_used:.2f}/{sys_total:.2f} GB ({vm.percent}%)")
    print("---------------------------\n")

def main():
    print_memory_usage("Start")
    
    ########################################
    # 1. Set hyperparameters + config
    ########################################
    # Use a smaller model for testing - CRITICAL for CPU testing
    use_smaller_model = False  # Set to False for actual training
    
    if use_smaller_model:
        model_name = "gpt2"  # ~124M parameters
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"  # ~3B parameters
    
    print(f"Using model: {model_name}")
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    
    # TESTING: Use extremely limited data and parameters
    stream_chunk_size = 2      # Only load 2 samples at a time
    micro_batch_size = 1       # One example per batch
    epochs_per_chunk = 1       # One epoch per chunk
    max_sequence_length = 128  # Very short sequences for testing
    learning_rate = 1e-4
    
    # Disable LoRA for initial testing
    use_lora = True
    
    output_dir = "saved_model_test"
    
    print_memory_usage("After config")
    
    ########################################
    # 2. Initialize experiment tracking (W&B)
    ########################################
    wandb.init(
        project="RealEstate-LLM-Finetuning-Test",
        name="CPU-test-run",
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size": micro_batch_size,
            "stream_chunk_size": stream_chunk_size,
        }
    )
    
    ########################################
    # 3. Load model + tokenizer
    ########################################
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    print("Tokenizer loaded successfully")
    
    print_memory_usage("After tokenizer")
    
    print("Loading model...")
    # Force CPU and FP32 for more stability
    model = load_model(
        model_name=model_name,
        use_lora=use_lora,
        lora_config=None,
        quantization_bits=0,   # No quantization for testing
        device_map="cpu"       # Force CPU
    )
    
    # Force model to CPU if it's not already
    model.to('cpu')
    
    # Disable features that increase memory usage
    model.config.use_cache = False
    model.eval()  # Start in eval mode until we're ready to train
    
    print("Model loaded successfully")
    print_memory_usage("After model loaded")
    
    # Test the model with a tiny input to verify it works
    print("Testing model with tiny input...")
    try:
        with torch.no_grad():
            test_ids = torch.tensor([[1, 2, 3, 4, 5]], device='cpu')
            _ = model(test_ids)
        print("Model test successful")
    except Exception as e:
        print(f"Model test failed: {e}")
        return  # Stop if basic model test fails
    
    ########################################
    # 4. Create optimizer + scheduler
    ########################################
    # If not using LoRA, we'll train just a few parameters
    if not use_lora:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Only unfreeze the output layer
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
    
    # Only optimize trainable parameters
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training with {trainable_params} trainable parameters")
    print_memory_usage("After optimizer")
    
    ########################################
    # 5. Stream data + training loop
    ########################################
    print("Setting up data processing...")
    preprocessor = Preprocessor()
    data_loader = CustomDataLoader(data_path)
    
    # We'll limit to just one chunk for testing
    max_chunks = 1
    chunk_count = 0
    
    # Prepare for training
    model.train()
    
    print("Starting data processing loop...")
    # We'll iterate over each chunk from the custom loader
    for raw_chunk in data_loader.stream_data(batch_size=stream_chunk_size):
        print(f"Processing chunk {chunk_count+1} with {len(raw_chunk)} samples")
        chunk_count += 1
        
        if chunk_count > max_chunks:
            print(f"Stopping after {max_chunks} chunks (for testing)")
            break
        
        # Limit to just 2 examples maximum for extreme testing
        if len(raw_chunk) > 2:
            print(f"Limiting chunk to 2 examples for testing")
            raw_chunk = raw_chunk[:2]
        
        print_memory_usage(f"After loading chunk {chunk_count}")
        
        # Preprocess chunk
        processed_examples = preprocessor.process_data(raw_chunk)
        print(f"Preprocessed {len(processed_examples)} examples")
        
        print_memory_usage("After preprocessing")
        
        # Create the dataset with limited sequence length
        dataset = RealEstateDataset.from_preprocessor(
            raw_data=raw_chunk,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            max_input_length=max_sequence_length,  # Very limited for testing
            max_output_length=max_sequence_length  # Very limited for testing
        )
        
        # Sanity check dataset
        if len(dataset) == 0:
            print("Warning: Dataset is empty after preprocessing")
            continue
            
        print(f"Created dataset with {len(dataset)} examples")
        print_memory_usage("After dataset creation")
        
        # Debug first example sizes
        example = dataset[0]
        print(f"First example - Input length: {len(example['input_ids'])}, Output included in labels length: {sum(1 for x in example['labels'] if x != -100)}")
        
        # Wrap in a PyTorch DataLoader
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=False,  # No shuffle for testing
            collate_fn=collate_fn
        )
        
        print(f"Created DataLoader with {len(train_loader)} batches")
        print_memory_usage("After DataLoader creation")
        
        # Train for X epochs on this chunk - just 1 for testing
        for epoch in range(epochs_per_chunk):
            print(f"Starting epoch {epoch+1}/{epochs_per_chunk}")
            
            # Process only 1 batch for extreme testing
            process_batches = 1
            
            for step, batch in enumerate(train_loader):
                # Skip after processing limited batches
                if step >= process_batches:
                    print(f"Stopping after {process_batches} batches (for testing)")
                    break
                    
                print(f"Processing batch {step+1}/{min(process_batches, len(train_loader))}")
                print_memory_usage(f"Start of batch {step+1}")
                
                try:
                    # Move data to device
                    input_ids = batch["input_ids"].to('cpu')
                    attention_mask = batch["attention_mask"].to('cpu')
                    labels = batch["labels"].to('cpu')
                    
                    # For ultra-safe testing, truncate sequences if still too long
                    if input_ids.size(1) > max_sequence_length:
                        print(f"Truncating batch from {input_ids.size(1)} to {max_sequence_length} tokens")
                        input_ids = input_ids[:, :max_sequence_length]
                        attention_mask = attention_mask[:, :max_sequence_length]
                        labels = labels[:, :max_sequence_length]
                    
                    print(f"Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    print(f"Forward pass complete - Loss: {loss.item():.4f}")
                    
                    print_memory_usage("After forward pass")
                    
                    # Backward pass
                    loss.backward()
                    print("Backward pass complete")
                    
                    print_memory_usage("After backward pass")
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    print_memory_usage("After optimizer step")
                    
                    # Log info
                    wandb.log({"loss": loss.item()})
                    
                    # Force garbage collection
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error during training step: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            print(f"Completed epoch {epoch+1}")
        
        print(f"Completed training on chunk {chunk_count}")
    
    ########################################
    # 6. Final model save + cleanup
    ########################################
    print("Training complete")
    print_memory_usage("End of training")
    
    # For testing, avoid saving the model
    print("Test complete - model not saved for testing")

if __name__ == "__main__":
    main()