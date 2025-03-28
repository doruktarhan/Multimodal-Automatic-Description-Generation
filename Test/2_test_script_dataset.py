import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer
from tqdm import tqdm

# Add path if needed
if os.path.exists('Data'):
    sys.path.append('.')
elif os.path.exists('../Data'):
    sys.path.append('..')

# Import custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.preprocessor import Preprocessor
from Data.dataset import RealEstateDataset, collate_fn


def test_dataset_implementation():
    """
    Test the dataset implementation with step-by-step verification
    """
    print("\n" + "="*80)
    print("TESTING DATASET IMPLEMENTATION")
    print("="*80)
    
    # 1. Load the data file
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    if not os.path.exists(data_path):
        data_path = "../Data/funda_scrapped_amsterdam_sample.json"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        print("Current working directory:", os.getcwd())
        return
    
    print(f"Loading data from: {data_path}")
    data_loader = CustomDataLoader(data_path)
    
    try:
        raw_data = data_loader.load_all()
        print(f"✓ Successfully loaded {len(raw_data)} samples")
    except Exception as e:
        print(f"× Error loading data: {e}")
        return
    
    # 2. Initialize tokenizer (with fallback)
    print("\nInitializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Successfully loaded Qwen tokenizer")
    except Exception as e:
        print(f"× Error loading Qwen tokenizer: {e}")
        print("Using GPT2 tokenizer as fallback...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = Preprocessor()
    
    # 4. Test preprocessor on sample data
    print("\nTesting preprocessor on first sample...")
    example = preprocessor.create_example(raw_data[0])
    if example:
        print(f"✓ Example created successfully")
        print(f"  Input begins with: '{example['input'][:50]}...'")
        print(f"  Output begins with: '{example['output'][:50]}...'")
        print(f"  Input length: {len(example['input'])} chars")
        print(f"  Output length: {len(example['output'])} chars")
    else:
        print("× Failed to create example from first sample")
    
    # 5. Process all data
    print("\nProcessing all examples...")
    examples = preprocessor.process_data(raw_data)
    print(f"✓ Processed {len(examples)} examples from {len(raw_data)} raw samples")
    
    # 6. Create dataset
    print("\nCreating dataset...")
    dataset = RealEstateDataset(
        examples, 
        tokenizer,
        #max_input_length=max_input_length,
        #max_output_length=max_output_length
    )
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # 7. Test single item retrieval
    print("\nTesting single item retrieval...")
    try:
        sample = dataset[0]
        print(f"✓ Successfully retrieved sample from dataset")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  labels shape: {sample['labels'].shape}")
        
        # Check if labels are correctly set (-100 for input part)
        num_input_tokens = (sample['labels'] == -100).sum().item()
        num_output_tokens = (sample['labels'] != -100).sum().item()
        print(f"  Number of input tokens (masked in labels): {num_input_tokens}")
        print(f"  Number of output tokens (used in loss): {num_output_tokens}")
        
        # Verify first and last tokens
        print(f"  First 5 input_ids: {sample['input_ids'][:5].tolist()}")
        print(f"  First 5 labels: {sample['labels'][:5].tolist()}")
        
        # Find transition point between input and output in labels
        transition_idx = None
        for i in range(len(sample['labels'])-1):
            if sample['labels'][i] == -100 and sample['labels'][i+1] != -100:
                transition_idx = i + 1
                break
                
        if transition_idx is not None:
            print(f"  Transition from input to output at index: {transition_idx}")
            print(f"  Tokens around transition: {sample['input_ids'][transition_idx-2:transition_idx+3].tolist()}")
            print(f"  Labels around transition: {sample['labels'][transition_idx-2:transition_idx+3].tolist()}")
    except Exception as e:
        print(f"× Error retrieving sample: {e}")
    
    # 8. Create DataLoader with dynamic padding
    print("\nTesting DataLoader with dynamic padding...")
    batch_size = min(4, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 9. Test batch processing
    print("\nProcessing a batch...")
    try:
        batch = next(iter(loader))
        print(f"✓ Successfully retrieved batch")
        print(f"  batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  batch labels shape: {batch['labels'].shape}")
        
        # Check padding
        seq_lengths = batch['attention_mask'].sum(dim=1)
        print(f"  Sequence lengths in this batch: {seq_lengths.tolist()}")
        padding_efficiency = seq_lengths.float().mean().item() / batch['input_ids'].shape[1] * 100
        print(f"  Padding efficiency: {padding_efficiency:.2f}% (higher is better)")
        
        # Check a couple of samples from the batch
        for i in range(min(2, batch_size)):
            print(f"\n  Sample {i+1} in batch:")
            input_length = (batch['labels'][i] == -100).sum().item()
            output_length = (batch['labels'][i] != -100).sum().item()
            print(f"    Input length: {input_length} tokens")
            print(f"    Output length: {output_length} tokens")
            print(f"    First 5 tokens: {batch['input_ids'][i][:5].tolist()}")
    except Exception as e:
        print(f"× Error processing batch: {e}")
    
    # 10. Process all batches and collect statistics
    print("\nProcessing all batches to collect statistics...")
    input_lengths = []
    output_lengths = []
    total_tokens = 0
    padding_tokens = 0
    
    for batch in tqdm(loader, desc="Processing batches"):
        # Get actual sequence lengths (non-padding tokens)
        actual_lengths = batch['attention_mask'].sum(dim=1)
        
        # Count input vs output tokens
        for i in range(batch['input_ids'].shape[0]):
            # Input tokens (where labels = -100)
            input_length = (batch['labels'][i] == -100).sum().item()
            input_lengths.append(input_length)
            
            # Output tokens (where labels != -100, excluding padding)
            output_length = ((batch['labels'][i] != -100) & (batch['attention_mask'][i] == 1)).sum().item()
            output_lengths.append(output_length)
            
            # Count padding tokens
            total_tokens += batch['input_ids'].shape[1]
            padding_tokens += (batch['attention_mask'][i] == 0).sum().item()
    
    # 11. Display statistics
    print("\nDataset statistics:")
    print(f"  Total examples processed: {len(input_lengths)}")
    print(f"  Average input length: {np.mean(input_lengths):.1f} tokens (min: {min(input_lengths)}, max: {max(input_lengths)})")
    print(f"  Average output length: {np.mean(output_lengths):.1f} tokens (min: {min(output_lengths)}, max: {max(output_lengths)})")
    padding_percentage = padding_tokens / total_tokens * 100
    print(f"  Padding efficiency: {100 - padding_percentage:.2f}% (higher is better)")
    
    # 12. Plot distribution of sequence lengths
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(input_lengths, bins=15, alpha=0.7, color='blue')
    plt.axvline(np.mean(input_lengths), color='red', linestyle='dashed', linewidth=1)
    plt.title('Input Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(output_lengths, bins=15, alpha=0.7, color='green')
    plt.axvline(np.mean(output_lengths), color='red', linestyle='dashed', linewidth=1)
    plt.title('Output Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = "token_length_distribution.png"
    plt.savefig(plot_path)
    print(f"\nToken length distribution plot saved to: {os.path.abspath(plot_path)}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Start the test
    test_dataset_implementation()