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
    load_model, load_tokenizer, create_lora_config, save_model
)
from Evaluation.train_eval_metrics import evaluate_model

#calculate token stats for a given pytorch dataset object
def get_token_statistics(dataset):
    input_lengths = []
    output_lengths = []
    for i in range(len(dataset)):
        item = dataset[i]
        output_length = sum(1 for x in item['labels'] if x != -100)
        input_length = len(item['input_ids']) - output_length
        input_lengths.append(input_length)
        output_lengths.append(output_length)
    
    return {
        "input_min": min(input_lengths),
        "input_max": max(input_lengths),
        "input_avg": sum(input_lengths)/len(input_lengths),
        "output_min": min(output_lengths),
        "output_max": max(output_lengths),
        "output_avg": sum(output_lengths)/len(output_lengths)
    }

def print_memory_usage(label):
    """Print GPU memory usage with a descriptive label"""
    if torch.cuda.is_available():
        # Get current GPU
        current_device = torch.cuda.current_device()
        
        # Memory reserved by PyTorch in bytes
        reserved = torch.cuda.memory_reserved(current_device) / (1024 * 1024)  # MB
        
        # Memory allocated by PyTorch in bytes
        allocated = torch.cuda.memory_allocated(current_device) / (1024 * 1024)  # MB
        
        # Get max memory allocated
        max_allocated = torch.cuda.max_memory_allocated(current_device) / (1024 * 1024)  # MB
        
        # Get total memory on GPU
        total = torch.cuda.get_device_properties(current_device).total_memory / (1024 * 1024)  # MB
        
        print(f"\n----- GPU MEMORY [{label}] -----")
        print(f"Device: {torch.cuda.get_device_name(current_device)}")
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")
        print(f"Max Allocated: {max_allocated:.2f} MB")
        print(f"Total GPU Memory: {total:.2f} MB")
        print(f"Memory Utilization: {allocated/total*100:.2f}%")
        print("---------------------------\n")
    else:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 * 1024)  # MB
        
        print(f"\n----- CPU MEMORY [{label}] -----")
        print(f"Process Memory: {rss:.2f} MB")
        print(f"Warning: CUDA not available, reporting CPU memory instead")
        print("---------------------------\n")

def get_default_device():
    """Get default device (CUDA if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    # Check if CUDA is available and print info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available with {device_count} device(s)")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    #print_memory_usage("Start")
    start_time = time.time()
    
    # Get default device
    default_device = get_default_device()
    print(f"Default device: {default_device}")
    
    ########################################
    # 1. Set hyperparameters + config
    ########################################
    # Select which model to use (~1B parameter models)
    model_options = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "phi": "microsoft/phi-1_5",
        "qwen": "Qwen/Qwen1.5-1.8B-Chat",
        "qwen3b" : "Qwen/Qwen2.5-3B-Instruct",
        "opt": "facebook/opt-1.3b",
        "gpt2": "gpt2"
    }
    
    selected_model = "qwen3b"  # Change this to select a different model
    model_name = model_options[selected_model]
    
    print(f"Using model: {model_name} ({selected_model})")


    data_path = "train_data.json"
    val_data_path = "val_data.json"


    # Training hyperparams - testing all 30 samples
    stream_chunk_size = 1024     # Load 10 samples at a time
    micro_batch_size = 2      # Train one sample at a time
    epochs = 3                   # Number of epochs
    max_sequence_length = 8192  # Moderate sequence length
    learning_rate = 5e-5       # Conservative learning rate
    quantization = 8            #select from [0,4,8] 0 for no quantization
    gradient_accumulation_steps = 8 # update parameters after every x steps for making effective learning rate desired 
    chunk_repeat_eval_save = 5 # save and evaluate model every x chunks 0 means no save and eval chunk base. 
    
    # For GPU, explicitly set device_map to "auto" which should use all available GPUs
    device_map = "auto"
    
    # Use LoRA for more efficient fine-tuning?
    use_lora = True
    
    if use_lora:
        print("Using LoRA for parameter-efficient fine-tuning")
        # Create a minimal LoRA config
        lora_cfg = create_lora_config(
            r=8,                # Low rank for memory efficiency
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
    else:
        print("Training selected layers without LoRA")
        lora_cfg = None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"saved_model_{selected_model}_bs{micro_batch_size}_lr{learning_rate}_{timestamp}"

    
    #print_memory_usage("After config")
    
    ########################################
    # 2. Initialize experiment tracking (W&B)
    ########################################
    wandb.init(
        project=f"LLM-RealEstate-{selected_model}-test",
        name=f"{selected_model}-training-test-{stream_chunk_size}-{micro_batch_size}-{learning_rate}-{timestamp}",
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
        quantization_bits=4,   # No quantization for testing
        device_map=device_map  # Use "auto" for automatic device placement
    )
    
    # Disable features that increase memory usage
    model.config.use_cache = False
    
    # Verify model is on GPU
    if torch.cuda.is_available():
        # Check where the model is located
        param_device = next(model.parameters()).device
        print(f"Model is on device: {param_device}")
        
        # If device_map is "auto", the model might be split across devices
        # So let's check a few key components
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                first_layer_device = next(model.model.layers[0].parameters()).device
                last_layer_device = next(model.model.layers[-1].parameters()).device
                print(f"First layer device: {first_layer_device}")
                print(f"Last layer device: {last_layer_device}")
    
    print("Model loaded successfully")
    print_memory_usage("After model loaded")
    
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
    
    #print_memory_usage("After optimizer")
    
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
    
    # Get model device for input tensors
    if torch.cuda.is_available():
        # If the model is split across devices, use the default device for inputs
        model_device = default_device
    else:
        model_device = torch.device('cpu')
    
    print(f"Using {model_device} for input tensors")
    
    print("Starting data processing and training loop...")

    # Track total processed examples
    total_examples = 0
    total_chunks = 0

    # First, count total chunks to better track progress
    print("Counting total chunks in dataset...")
    for _ in data_loader.stream_data(batch_size=stream_chunk_size):
        total_chunks += 1
    print(f"Total chunks in dataset: {total_chunks}")

    ############### Validation Data Creation #########################

    val_raw_data = []
    try:
        val_data_loader = CustomDataLoader(val_data_path)
        # Load validation data
        for batch in val_data_loader.stream_data(batch_size=100):
            val_raw_data.extend(batch)

        val_processed = preprocessor.process_data(val_raw_data)
        print(f"Preprocessed {len(val_processed)} validation examples")
        val_dataset = RealEstateDataset.from_preprocessor(
            raw_data=val_raw_data,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            max_input_length=max_sequence_length,
            max_output_length=max_sequence_length
        )
        print(f"Created validation dataset with {len(val_dataset)} examples")

        #track best metrics for validation
        best_metrics = {'perplexity': float('inf'), 'bleu_4': 0, 'rougeL': 0}
        best_model_path = os.path.join(output_dir, "best_model")
        
    except: 
        print(f"No validation data exists")

    ############################# Training loop ###################################

    # Train for the specified number of epochs
    for epoch in range(epochs):  # Now this variable means epochs over all data
        print(f"\n===== EPOCH {epoch+1}/{epochs} =====")
        epoch_loss = 0.0
        epoch_batches = 0
        chunk_count = 0
        
        # Reset for each epoch
        data_loader = CustomDataLoader(data_path)  # Reinitialize the data loader
        
        # Process each chunk once per epoch
        for raw_chunk in data_loader.stream_data(batch_size=stream_chunk_size):
            chunk_size = len(raw_chunk)
            chunk_count += 1
            total_examples += chunk_size
            
            print(f"\n----- Processing Chunk {chunk_count}/{total_chunks} in Epoch {epoch+1} -----")
            print(f"Chunk size: {chunk_size} samples")
            
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
            
            #get the token stats and print them
            token_stats = get_token_statistics(dataset)
            print(f"Input token statistics - Min: {token_stats['input_min']}, Max: {token_stats['input_max']}, Avg: {token_stats['input_avg']}")
            print(f"Output token statistics - Min: {token_stats['output_min']}, Max: {token_stats['output_max']}, Avg: {token_stats['output_avg']:.1f}")
            
            # Create DataLoader
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=micro_batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            print(f"Created DataLoader with {len(train_loader)} batches")
            
            # Initialize optimizer at the start of processing
            optimizer.zero_grad()
            
            # Process all batches in this chunk
            for step, batch in enumerate(train_loader):
                print(f"Training batch {step+1}/{len(train_loader)} (Chunk {chunk_count}/{total_chunks}, Epoch {epoch+1})")
                
                try:
                    # Get timing information for GPU operations
                    start_batch_time = time.time()
                    
                    # Move data to device (either GPU or CPU)
                    input_ids = batch["input_ids"].to(model_device)
                    attention_mask = batch["attention_mask"].to(model_device)
                    labels = batch["labels"].to(model_device)
                    
                    # Print sequence length for monitoring
                    seq_len = input_ids.size(1)
                    print(f"Sequence length: {seq_len} tokens")
                    
                    # Catch sequences that are too long
                    if seq_len > 4096:
                        print(f"Warning: Sequence length {seq_len} is very long, might cause memory issues")
                    
                    # Forward pass
                    forward_start = time.time()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = outputs.loss / gradient_accumulation_steps
                    forward_time = time.time() - forward_start
                    print(f"Forward pass took {forward_time:.2f} seconds")
                    
                    print(f"Loss: {loss.item():.4f}")
                    epoch_loss += loss.item() * gradient_accumulation_steps  # Record unscaled loss for reporting
                    epoch_batches += 1
                    
                    # Backward pass
                    backward_start = time.time()
                    loss.backward()
                    backward_time = time.time() - backward_start
                    print(f"Backward pass took {backward_time:.2f} seconds")
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 
                        max_norm=1.0
                    )
                    
                    # Update weights if gradient accumulation step is reached
                    if (epoch_batches % gradient_accumulation_steps == 0):
                        optimizer.step()    
                        optimizer.zero_grad()
                    
                    # Total batch time
                    batch_time = time.time() - start_batch_time
                    print(f"Batch processing took {batch_time:.2f} seconds total")
                    
                    # Log metrics
                    wandb.log({
                        "batch_loss": loss.item() * gradient_accumulation_steps,  # Log unscaled loss
                        "sequence_length": seq_len,
                        "forward_time": forward_time,
                        "backward_time": backward_time,
                        "batch_time": batch_time,
                        "global_step": epoch_batches,
                        "epoch": epoch,
                        "chunk": chunk_count
                    })
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue  # Try to continue with next batch
                    
            # Save checkpoint after each chunk if needed
            if chunk_count % chunk_repeat_eval_save == 0:  # Save every 5 chunks
                checkpoint_dir = os.path.join(output_dir, f"epoch{epoch+1}_chunk{chunk_count}")
                try:
                    print(f"Saving intermediate checkpoint to {checkpoint_dir}")
                    save_model(model, checkpoint_dir)
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
            
            # evaluate the checkpointed model if validation data exists
            if chunk_count % chunk_repeat_eval_save == 0 and val_raw_data is not None:
                print(f"Evaluating model on validation set for  epoch {epoch+1} chunk {chunk_count}/{total_chunks}")
                metrics,examples = evaluate_model(
                    model=model,
                    val_dataset=val_dataset,
                    tokenizer=tokenizer,
                    model_device=model_device,
                    batch_size=micro_batch_size,
                    max_gen_length=max_sequence_length,
                    num_samples=None,  # Evaluate all samples
                    collate_fn=collate_fn
                )
                wandb.log({
                    "val_perplexity": metrics['perplexity'],
                    "val_bleu_1": metrics['bleu_1'],
                    "val_bleu_4": metrics['bleu_4'],
                    "val_meteor": metrics['meteor'],
                    "val_rouge1": metrics['rouge1'],
                    "val_rouge2": metrics['rouge2'],
                    "val_rougeL": metrics['rougeL'],
                    "epoch": epoch,
                    "chunk": chunk_count,
                    "global_step": epoch_batches
                })
                        # Check if this is the best model so far
                is_best = False
                if metrics['perplexity'] < best_metrics['perplexity']:
                    best_metrics['perplexity'] = metrics['perplexity']
                    is_best = True
                    
                if metrics['bleu_4'] > best_metrics['bleu_4']:
                    best_metrics['bleu_4'] = metrics['bleu_4']
                    is_best = True
                    
                if metrics['rougeL'] > best_metrics['rougeL']:
                    best_metrics['rougeL'] = metrics['rougeL']
                    is_best = True

                # Save the best model
                if is_best:
                    try:
                        print(f"New best model! Saving to {best_model_path}")
                        save_model(model, best_model_path)
                        
                        # Save the metrics alongside the model
                        with open(os.path.join(best_model_path, "metrics.txt"), "w") as f:
                            for k, v in metrics.items():
                                f.write(f"{k}: {v:.4f}\n")
                    except Exception as e:
                        print(f"Failed to save best model: {e}")



        
        # Handle any remaining gradients at the end of all chunks
        if epoch_batches % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()


        ###################### Evaluation Step After Epoch #################################
        # Evaluate model on validation set (if available)
        # evaluate the checkpointed model if validation data exists
        if chunk_count % chunk_repeat_eval_save == 0 and val_raw_data is not None:
            print(f"Evaluating model on validation set for  epoch {epoch+1} chunk {chunk_count}/{total_chunks}")
            metrics,examples = evaluate_model(
                model=model,
                val_dataset=val_dataset,
                tokenizer=tokenizer,
                model_device=model_device,
                batch_size=micro_batch_size,
                max_gen_length=max_sequence_length,
                num_samples=None,  # Evaluate all samples
                collate_fn=collate_fn
            )
            wandb.log({
                "val_perplexity": metrics['perplexity'],
                "val_bleu_1": metrics['bleu_1'],
                "val_bleu_4": metrics['bleu_4'],
                "val_meteor": metrics['meteor'],
                "val_rouge1": metrics['rouge1'],
                "val_rouge2": metrics['rouge2'],
                "val_rougeL": metrics['rougeL'],
                "epoch": epoch,
                "chunk": chunk_count,
                "global_step": epoch_batches
            })
                    # Check if this is the best model so far
            is_best = False
            if metrics['perplexity'] < best_metrics['perplexity']:
                best_metrics['perplexity'] = metrics['perplexity']
                is_best = True
                
            if metrics['bleu_4'] > best_metrics['bleu_4']:
                best_metrics['bleu_4'] = metrics['bleu_4']
                is_best = True
                
            if metrics['rougeL'] > best_metrics['rougeL']:
                best_metrics['rougeL'] = metrics['rougeL']
                is_best = True

            # Save the best model
            if is_best:
                try:
                    print(f"New best model! Saving to {best_model_path}")
                    save_model(model, best_model_path)
                    
                    # Save the metrics alongside the model
                    with open(os.path.join(best_model_path, "metrics.txt"), "w") as f:
                        for k, v in metrics.items():
                            f.write(f"{k}: {v:.4f}\n")
                except Exception as e:
                    print(f"Failed to save best model: {e}")

            
        # Report epoch stats
        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            print(f"\nEpoch {epoch+1} complete - Average loss: {avg_loss:.4f}")
            print(f"Processed {epoch_batches} batches from {chunk_count} chunks")
            wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch, "completed_chunks": chunk_count})
        
        # Save model after each epoch
        epoch_dir = os.path.join(output_dir, f"epoch{epoch+1}_full")
        try:
            print(f"Saving model after epoch {epoch+1} to {epoch_dir}")
            save_model(model, epoch_dir)
        except Exception as e:
            print(f"Failed to save epoch model: {e}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()