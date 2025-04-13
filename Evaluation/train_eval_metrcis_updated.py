import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
import math
import numpy as np
import wandb
from tqdm import tqdm
import gc
import time


# Download NLTK resources if needed
nltk.download('punkt_tab')
nltk.download('wordnet')


def evaluate_model(model, tokenizer, val_dataset, model_device, collate_fn, batch_size=4, max_gen_length=800, num_samples=8):
    """
    Evaluate model on validation dataset
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        val_dataset: Validation dataset
        model_device: Device to run evaluation on
        collate_fn: Collate function to use for the dataset
        batch_size: Batch size for evaluation
        max_gen_length: Maximum number of new tokens to generate
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dictionary of metrics and qualitative examples
    """
    eval_start_time = time.time()

    model.eval()  # Set model to evaluation mode
    
    # Create dataloader for validation set
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize metrics
    total_loss = 0
    total_tokens = 0
    all_refs = []
    all_hyps = []
    
    # Limit number of samples if specified
    if num_samples is not None:
        sample_batches = min(num_samples // batch_size + 1, len(val_loader))
    else:
        sample_batches = len(val_loader)
    
    # ROUGE scorer and BLEU smoothing
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    
    print(f"Evaluating model on {sample_batches} batches...")
    
    # Counter for total generated examples for logging purposes
    generated_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=sample_batches)):
            if i >= sample_batches:
                break
                
            # Move data to device
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
            labels = batch["labels"].to(model_device)
            
            # Calculate perplexity via teacher forcing
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item() * torch.sum(labels != -100).item()
            total_tokens += torch.sum(labels != -100).item()
            
            # Process each sample in the batch for generation
            for j in range(input_ids.size(0)):
                generated_count += 1
                
                # Get input text (prompt). We assume tokens with label -100 are not generated (i.e., part of the prompt)
                prompt_length = torch.sum(batch["labels"][j] == -100).item()
                input_text_ids = input_ids[j, :prompt_length]
                
                # Get reference (ground truth) text
                ref_text_ids = labels[j, labels[j] != -100]
                ref_text = tokenizer.decode(ref_text_ids, skip_special_tokens=True)
                all_refs.append(ref_text)
                
                # Prepare generation input by adding the batch dimension
                gen_input = input_text_ids.unsqueeze(0)
                
                # Time the generation process for this sample
                gen_start = time.time()
                output_ids = model.generate(
                    gen_input.to(model_device),
                    max_new_tokens=max_gen_length,  # Limit the number of new tokens
                    do_sample=False,                # Use deterministic (greedy) decoding
                    num_beams=1                     # Greedy decoding; no beam search overhead
                )
                gen_end = time.time()
                gen_time = gen_end - gen_start
                
                # Remove the prompt part from the generated output
                gen_text_ids = output_ids[0, prompt_length:]
                gen_text = tokenizer.decode(gen_text_ids, skip_special_tokens=True)
                all_hyps.append(gen_text)
                
                # Print generation details for this example
                generated_length = gen_text_ids.size(0)
                print(f"Generated sample {generated_count}: generated in {gen_time:.2f} seconds; output length: {generated_length} tokens")
    
    # Compute perplexity
    if total_tokens > 0:
        perplexity = math.exp(min(total_loss / total_tokens, 20))  # Cap exponent to avoid overflow
    else:
        perplexity = float('inf')
    
    # Compute BLEU, METEOR, and ROUGE metrics
    bleu_1 = 0.0
    bleu_4 = 0.0
    meteor_scores = 0.0
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for ref, hyp in zip(all_refs, all_hyps):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        
        if len(hyp_tokens) > 0:
            bleu_1 += sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu_4 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            meteor_scores += meteor_score([ref_tokens], hyp_tokens)
        
        rouge_result = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key] += rouge_result[key].fmeasure
    
    num_eval_examples = len(all_refs)
    if num_eval_examples > 0:
        bleu_1 /= num_eval_examples
        bleu_4 /= num_eval_examples
        meteor_scores /= num_eval_examples
        for key in rouge_scores:
            rouge_scores[key] /= num_eval_examples
    
    # Compile metrics
    metrics = {
        'perplexity': perplexity,
        'bleu_1': bleu_1,
        'bleu_4': bleu_4,
        'meteor': meteor_scores,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
    }
    
    # Generate a few examples for qualitative evaluation
    examples = []
    for i in range(min(3, len(all_refs))):
        examples.append({
            'reference': all_refs[i],
            'generated': all_hyps[i]
        })
    
    # Log examples to wandb if available
    if wandb.run is not None:
        examples_table = wandb.Table(columns=["Reference", "Generated"])
        for example in examples:
            examples_table.add_data(example['reference'], example['generated'])
        wandb.log({"generation_examples": examples_table})
    
    # Print examples and metrics
    print("\n=== EVALUATION EXAMPLES ===")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Reference: {example['reference'][:100]}...")
        print(f"Generated: {example['generated'][:100]}...")
    
    print("\n=== EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    model.train()  
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()              # Trigger garbage collection

    eval_end_time = time.time()
    total_eval_time = eval_end_time - eval_start_time

    # Print overall evaluation time message
    if num_samples is not None:
        print(f"Evaluation for {num_samples} out of {len(val_dataset)} samples processed in {total_eval_time:.2f} seconds")
    else:
        print(f"Full evaluation for {len(val_dataset)} samples processed in {total_eval_time:.2f} seconds")
    
    return metrics, examples
