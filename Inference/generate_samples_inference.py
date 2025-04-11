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
from Data.dataset import RealEstateDataset, collate_fn
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
        
    Returns:
        Dictionary mapping property IDs to generated descriptions
    """
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if verbose:
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
    
    # Extract property IDs for reference
    property_ids = []
    for item in test_data:
        prop_id = item.get("id", item.get("property_name", f"unknown_{len(property_ids)}"))
        property_ids.append(prop_id)
    
    # Create dataset
    if verbose:
        print("Creating dataset from test data...")
    
    start_dataset_time = time.time()
    dataset = RealEstateDataset.from_preprocessor(
        raw_data=test_data,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        max_input_length=max_length,
        max_output_length=max_length
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
    
    # Prepare results dictionary
    results = {}
    
    # Track generation metrics
    start_time = time.time()
    total_tokens_generated = 0
    batch_times = []
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating", disable=not verbose)):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(property_ids))
        batch_property_ids = property_ids[batch_start_idx:batch_end_idx]
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Extract input lengths for later separating input from output
        input_lengths = []
        for i in range(input_ids.size(0)):
            # Find where labels stop being -100 (that's where input ends)
            labels = batch["labels"][i]
            input_length = sum(1 for x in labels if x == -100)
            input_lengths.append(input_length)
        
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
        
        # Clear GPU cache between batches
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate overall performance
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
    
    perf_stats = {
        "total_samples": len(test_data),
        "total_tokens_generated": total_tokens_generated,
        "total_time_seconds": total_time,
        "tokens_per_second": avg_tokens_per_sec,
        "average_batch_time": sum(batch_times) / len(batch_times) if batch_times else 0,
        "batch_size": batch_size
    }
    
    if verbose:
        print(f"\nGeneration summary:")
        print(f"- Generated descriptions for {len(results)}/{len(test_data)} samples")
        print(f"- Total generation time: {total_time:.2f}s")
        print(f"- Average speed: {avg_tokens_per_sec:.2f} tokens/second")
        print(f"- Average batch time: {perf_stats['average_batch_time']:.2f}s")
    
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

def main(
    base_model_name: str,
    test_data_path: str,
    trained_model_path: Optional[str] = None,
    output_path: str = "generated_descriptions.json",
    batch_size: int = 4,
    max_length: int = 1024,
    quantization_bits: int = 8,
    temperature: float = 0.7,
):
    """
    Main function to load model and generate descriptions.
    
    Args:
        base_model_name: Base model name or path
        test_data_path: Path to test data
        trained_model_path: Path to trained model (if None, uses base model)
        output_path: Path to save the generated descriptions
        batch_size: Batch size for generation
        max_length: Maximum generation length
        quantization_bits: Quantization bits (0, 4, or 8)
        temperature: Temperature for generation
    """
    from Model.model_utils import load_trained_model
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    loader = CustomDataLoader(test_data_path)
    test_data = loader.load_all()
    print(f"Loaded {len(test_data)} test samples")
    
    # Create preprocessor
    preprocessor = Preprocessor()
    
    # Load model and tokenizer
    model, tokenizer = load_trained_model(
        base_model_name=base_model_name,
        trained_model_path=trained_model_path,
        quantization_bits=quantization_bits
    )
    
    # Generate descriptions
    results = generate_samples(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        preprocessor=preprocessor,
        batch_size=batch_size,
        max_length=max_length,
        temperature=temperature,
        output_path=output_path
    )
    
    print("Inference completed successfully!")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate real estate descriptions")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to trained model (if None, uses base model)")
    parser.add_argument("--output_path", type=str, default="generated_descriptions.json",
                        help="Path to save generated descriptions")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum generation length")
    parser.add_argument("--quantization", type=int, default=8, choices=[0, 4, 8],
                        help="Quantization bits (0, 4, or 8)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation sampling")
    
    args = parser.parse_args()
    
    main(
        base_model_name=args.base_model,
        test_data_path=args.test_data,
        trained_model_path=args.trained_model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        quantization_bits=args.quantization,
        temperature=args.temperature
    )