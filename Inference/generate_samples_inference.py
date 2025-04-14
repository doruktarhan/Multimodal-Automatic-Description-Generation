import os
import sys
import time
import json
import torch
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.preprocessor import Preprocessor
from Data.dataset import RealEstateDatasetForInference, collate_fn
from Model.model_utils import load_tokenizer


def generate_samples(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    preprocessor: Preprocessor,
    batch_size: int = 4,
    max_length: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 10,
    num_beams: int = 1,
    do_sample: bool = True,
    device: str = "cuda",
    output_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
    early_stopping: bool = True,
    test_mode: bool = False,  # New parameter for test mode
    test_batches: int = 2,    # Number of batches to process in test mode
):
    """
    Generate descriptions for a list of test samples using a trained model.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer for the model
        test_data: List of test data samples
        preprocessor: Preprocessor to create prompts
        batch_size: Size of batches for inference
        max_length: Maximum generation length
        temperature: Temperature for generation sampling
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search (1 = greedy/sampling)
        do_sample: Whether to use sampling (True) or greedy generation (False)
        device: Device to run inference on ("cuda" or "cpu")
        output_path: Path to save the results (if None, doesn't save)
        seed: Random seed for reproducibility
        verbose: Whether to print progress and timing information
        early_stopping: Whether to stop generation early on EOS token
        test_mode: Whether to run in test mode (process only a few batches)
        test_batches: Number of batches to process in test mode
        
    Returns:
        Dictionary mapping property IDs to generated descriptions
    """
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if verbose:
        if test_mode:
            print(f"Running in TEST MODE - will only process {test_batches} batches")
        print(f"Starting generation for {len(test_data)} samples with batch size {batch_size}")
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    if verbose:
        print(f"Using device: {device}")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # In test mode, limit the number of samples to process
    if test_mode:
        samples_to_process = min(test_batches * batch_size, len(test_data))
        if verbose:
            print(f"Test mode: Processing only {samples_to_process} samples")
        test_data = test_data[:samples_to_process]
    
    # Extract property IDs for reference
    property_ids = []
    for item in test_data:
        prop_id = item.get("id", item.get("property_name", f"unknown_{len(property_ids)}"))
        property_ids.append(prop_id)
    
    # Create dataset
    if verbose:
        print("Creating dataset from test data...")
    
    start_dataset_time = time.time()
    dataset = RealEstateDatasetForInference.from_preprocessor(
        raw_data=test_data,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        max_input_length=max_length
    )
    
    if verbose:
        dataset_time = time.time() - start_dataset_time
        print(f"Created dataset with {len(dataset)} examples in {dataset_time:.2f}s")
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for mapping back to property IDs
        collate_fn=collate_fn
    )
    
    if verbose:
        print(f"Created DataLoader with {len(dataloader)} batches")
        if test_mode:
            print(f"In test mode: will process only {min(test_batches, len(dataloader))} batches")
    
    # Prepare results dictionary
    results = {}
    
    # Track generation metrics
    start_time = time.time()
    total_tokens_generated = 0
    batch_times = []
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating", disable=not verbose)):
        # In test mode, process only the specified number of batches
        if test_mode and batch_idx >= test_batches:
            if verbose:
                print(f"Test mode: Stopping after {test_batches} batches")
            break
            
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(property_ids))
        batch_property_ids = property_ids[batch_start_idx:batch_end_idx]
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # We simply track the input length for each example to know where generation begins
        input_lengths = [len(ids) for ids in batch["input_ids"]]
        

        
        # Generate outputs
        batch_start_time = time.time()
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Calculate tokens per second for this batch
                total_input_tokens = input_ids.numel()
                total_output_tokens = outputs.numel()
                new_tokens = total_output_tokens - total_input_tokens
                
                total_tokens_generated += new_tokens
                
                tokens_per_sec = new_tokens / batch_time if batch_time > 0 else 0
                if verbose:
                    print(f"Batch {batch_idx+1}/{len(dataloader)}: {new_tokens} tokens in {batch_time:.2f}s ({tokens_per_sec:.2f} t/s)")
                
            except Exception as e:
                if verbose:
                    print(f"Error during generation for batch {batch_idx+1}: {e}")
                # Return empty results for failed samples
                for j, prop_id in enumerate(batch_property_ids):
                    if prop_id not in results:  # Avoid overwriting existing results
                        results[prop_id] = f"Generation failed: {str(e)}"
                continue
        
        # Process each sample in batch
        for j, (output, input_length, prop_id) in enumerate(zip(outputs, input_lengths, batch_property_ids)):
            if j >= len(batch_property_ids):
                break  # Safety check
                
            # Extract only the generated part (skip input)
            generated_part = output[input_length:]
            
            # Decode the text
            text = tokenizer.decode(generated_part, skip_special_tokens=True)
            
            # Store in results
            results[prop_id] = text.strip()
            
            # In test mode, print the first few characters of the result
            if test_mode and verbose:
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"\nSample output for {prop_id}:\n{preview}")
        
        # Clear GPU cache between batches
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate overall performance
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
    
    perf_stats = {
        "total_samples": len(results),
        "total_tokens_generated": total_tokens_generated,
        "total_time_seconds": total_time,
        "tokens_per_second": avg_tokens_per_sec,
        "average_batch_time": sum(batch_times) / len(batch_times) if batch_times else 0,
        "batch_size": batch_size,
        "test_mode": test_mode
    }
    
    if verbose:
        print(f"\nGeneration summary:")
        print(f"- Generated descriptions for {len(results)}/{len(test_data)} samples")
        print(f"- Total generation time: {total_time:.2f}s")
        print(f"- Average speed: {avg_tokens_per_sec:.2f} tokens/second")
        print(f"- Average batch time: {perf_stats['average_batch_time']:.2f}s")
        if test_mode:
            print(f"- TEST MODE was enabled - only processed {min(test_batches, len(dataloader))} batches")
    
    # Save results if output path is provided
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save results with performance stats
        output_data = {
            "results": results,
            "performance": perf_stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"Results saved to {output_path}")
    
    return results