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
    start_time = time.time()
    
    ########################################
    # 1. Set hyperparameters + config
    ########################################
    # Select which model to use (~1B parameter models)
    model_options = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "phi": "microsoft/phi-1_5",
        "qwen": "Qwen/Qwen1.5-1.8B-Chat",
        "opt": "facebook/opt-1.3b"
    }
    
    selected_model = "tinyllama"  # Change this to select a different model
    model_name = model_options[selected_model]
    
    print(f"Using model: {model_name} ({selected_model})")
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    
    # Training hyperparams - testing all 30 samples
    stream_chunk_size = 10     # Load 10 samples at a time
    micro_batch_size = 1       # Train one sample at a time
    epochs_per_chunk = 1       # One epoch per chunk
    max_sequence_length = 512  # Moderate sequence length
    learning_rate = 5e-5       # Conservative learning rate
    
    # Use LoRA for more efficient fine-tuning?
    use_lora = True
    
    if use_lora:
        print("Using LoRA for parameter-efficient fine-tuning")
        # Create a minimal LoRA config
        lora_cfg = create_lora_config(
            r=4,                # Low rank for memory efficiency
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[    # Target only key modules
                "q_proj", "v_proj"
            ]
        )
    else:
        print("Training selected layers without LoRA")
        lora_cfg = None
    
    output_dir = "saved_model_test"
    
    print_memory_usage("After config")
    
    ########################################
    # 2. Initialize experiment tracking (W&B)
    ########################################
    wandb.init(
        project=f"LLM-RealEstate-{selected_model}-test",
        name=f"{selected_model}-training-test",
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size": micro_batch_size,
            "stream_chunk_size": stream_chunk_size,
            "use_lora": use_lora
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
    model = load_model(
        model_name=model_name,
        use_lora=use_lora,
        lora_config=lora_cfg,
        quantization_bits=0,   # No quantization for testing
        device_map="cpu"       # Force CPU
    )
    
    # Disable features that increase memory usage
    model.config.use_cache = False
    
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
    # 4. Create optimizer
    ########################################
    # If not using LoRA and also not training the full model
    if not use_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Then unfreeze specific layers
        if hasattr(model, "lm_head"):
            # Option 1: Just train the language model head
            for param in model.lm_head.parameters():
                param.requires_grad = True
                print("Training LM head layer")
        else:
            # Option 2: Train the last transformer layer
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                last_layer = model.model.layers[-1]
                for param in last_layer.parameters():
                    param.requires_grad = True
                print("Training last transformer layer")
    
    # Create optimizer with trainable parameters only
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Training with {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    print(f"Percentage of parameters being trained: {trainable_params/total_params*100:.2f}%")
    
    print_memory_usage("After optimizer")
    
    ########################################
    # 5. Stream data + training loop
    ########################################
    print("Setting up data processing...")
    preprocessor = Preprocessor()
    data_loader = CustomDataLoader(data_path)
    
    # Track total processed examples
    total_examples = 0
    chunk_count = 0
    
    # Prepare for training
    model.train()
    
    print("Starting data processing and training loop...")
    
    # We'll iterate over each chunk from the custom loader
    for raw_chunk in data_loader.stream_data(batch_size=stream_chunk_size):
        chunk_size = len(raw_chunk)
        chunk_count += 1
        total_examples += chunk_size
        
        print(f"\n===== PROCESSING CHUNK {chunk_count} WITH {chunk_size} SAMPLES =====")
        print(f"Total examples processed so far: {total_examples}")
        
        print_memory_usage(f"After loading chunk {chunk_count}")
        
        # Preprocess chunk
        processed_examples = preprocessor.process_data(raw_chunk)
        print(f"Preprocessed {len(processed_examples)} examples")
        
        # Create the dataset
        dataset = RealEstateDataset.from_preprocessor(
            raw_data=raw_chunk,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            max_input_length=max_sequence_length,
            max_output_length=max_sequence_length
        )
        
        print(f"Created dataset with {len(dataset)} examples")
        
        # Get token statistics for the dataset
        input_lengths = []
        output_lengths = []
        for i in range(len(dataset)):
            item = dataset[i]
            # Count tokens that are not -100 in labels (these are output tokens)
            output_length = sum(1 for x in item['labels'] if x != -100)
            # Total length minus output length is input length
            input_length = len(item['input_ids']) - output_length
            input_lengths.append(input_length)
            output_lengths.append(output_length)
        
        print(f"Input token statistics - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.1f}")
        print(f"Output token statistics - Min: {min(output_lengths)}, Max: {max(output_lengths)}, Avg: {sum(output_lengths)/len(output_lengths):.1f}")
        
        print_memory_usage("After dataset creation")
        
        # Create DataLoader with batch size 1
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,        # Shuffle for better training
            collate_fn=collate_fn
        )
        
        print(f"Created DataLoader with {len(train_loader)} batches")
        
        # Train for specified epochs on this chunk
        for epoch in range(epochs_per_chunk):
            print(f"\n----- EPOCH {epoch+1}/{epochs_per_chunk} -----")
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_loader):
                print(f"Training batch {step+1}/{len(train_loader)}")
                
                try:
                    # Move data to device
                    input_ids = batch["input_ids"].to('cpu')
                    attention_mask = batch["attention_mask"].to('cpu')
                    labels = batch["labels"].to('cpu')
                    
                    # Print sequence length for monitoring
                    seq_len = input_ids.size(1)
                    print(f"Sequence length: {seq_len} tokens")
                    
                    # Catch sequences that are too long
                    if seq_len > 1024:  # Arbitrary safety limit
                        print(f"Warning: Sequence length {seq_len} is very long, might cause memory issues")
                    
                    # Forward pass with gradient accumulation
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        print("Warning: NaN loss detected, skipping batch")
                        continue
                        
                    print(f"Loss: {loss.item():.4f}")
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 
                        max_norm=1.0
                    )
                    
                    # Update weights
                    optimizer.step()
                    
                    # Log metrics
                    wandb.log({
                        "batch_loss": loss.item(),
                        "sequence_length": seq_len,
                        "batch": step,
                        "epoch": epoch,
                        "chunk": chunk_count
                    })
                    
                    # Force memory cleanup
                    gc.collect()
                    print_memory_usage(f"After batch {step+1}")
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue  # Try to continue with next batch
            
            # Report epoch stats
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1} complete - Average loss: {avg_loss:.4f}")
                wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch, "chunk": chunk_count})
        
        print(f"Completed training on chunk {chunk_count}")
        
        # Save intermediate model after each chunk
        if chunk_count % 2 == 0:  # Save every 2 chunks
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_chunk_{chunk_count}")
            try:
                print(f"Saving intermediate checkpoint to {checkpoint_dir}")
                save_model(model, checkpoint_dir)
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
    
    ########################################
    # 6. Final model save + cleanup
    ########################################
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n===== TRAINING COMPLETE =====")
    print(f"Processed {total_examples} examples in {chunk_count} chunks")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print_memory_usage("End of training")
    
    # Save final model
    try:
        print(f"Saving final model to {output_dir}")
        save_model(model, output_dir)
        print("Model saved successfully")
    except Exception as e:
        print(f"Failed to save model: {e}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()