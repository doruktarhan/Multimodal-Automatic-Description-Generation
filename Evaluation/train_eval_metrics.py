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


# Download NLTK resources if needed
nltk.download('punkt_tab')
nltk.download('wordnet')


def evaluate_model(model, tokenizer, val_dataset, model_device,collate_fn, batch_size=4, max_gen_length=2000, num_samples=None ):
    """
    Evaluate model on validation dataset
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        val_dataset: Validation dataset
        model_device: Device to run evaluation on
        batch_size: Batch size for evaluation
        max_gen_length: Maximum length for text generation
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dictionary of metrics
    """
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
    
    # ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # BLEU smoothing
    smoothie = SmoothingFunction().method1
    
    print(f"Evaluating model on {sample_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=sample_batches)):
            if i >= sample_batches:
                break
                
            # Move data to device
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
            labels = batch["labels"].to(model_device)
            
            # Calculate perplexity
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Keep track of loss and number of tokens for perplexity calculation
            total_loss += loss.item() * torch.sum(labels != -100).item()
            total_tokens += torch.sum(labels != -100).item()
            
            # For each example in batch, get reference text and generate prediction
            for j in range(input_ids.size(0)):
                # Get input text (prompt)
                prompt_length = torch.sum(batch["labels"][j] == -100).item()
                input_text_ids = input_ids[j, :prompt_length]
                
                # Get reference (ground truth) text
                ref_text_ids = labels[j, labels[j] != -100]
                ref_text = tokenizer.decode(ref_text_ids, skip_special_tokens=True)
                all_refs.append(ref_text)
                
                # Generate text
                gen_input = input_text_ids.unsqueeze(0)  # Add batch dimension
                
                # Generate text with the model
                output_ids = model.generate(
                    gen_input.to(model_device),
                    max_length=prompt_length + max_gen_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )
                
                # Decode generated text (remove prompt)
                gen_text_ids = output_ids[0, prompt_length:]
                gen_text = tokenizer.decode(gen_text_ids, skip_special_tokens=True)
                all_hyps.append(gen_text)
    
    # Calculate metrics
    if total_tokens > 0:
        # Cap the exponent to prevent overflow
        perplexity = math.exp(min(total_loss / total_tokens, 20))  # e^20 for nan values
    else:
        perplexity = float('inf')
    
    # Calculate BLEU scores
    bleu_1 = 0.0
    bleu_4 = 0.0
    meteor_scores = 0.0
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for ref, hyp in zip(all_refs, all_hyps):
        # Tokenize
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        
        # BLEU
        if len(hyp_tokens) > 0:
            bleu_1 += sentence_bleu([ref_tokens], hyp_tokens, 
                                    weights=(1, 0, 0, 0), 
                                    smoothing_function=smoothie)
            bleu_4 += sentence_bleu([ref_tokens], hyp_tokens, 
                                    weights=(0.25, 0.25, 0.25, 0.25), 
                                    smoothing_function=smoothie)
        
        # METEOR
        if len(hyp_tokens) > 0:
            meteor_scores += meteor_score([ref_tokens], hyp_tokens)
        
        # ROUGE
        rouge_result = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key] += rouge_result[key].fmeasure
    
    # Average the scores
    num_samples = len(all_refs)
    if num_samples > 0:
        bleu_1 /= num_samples
        bleu_4 /= num_samples
        meteor_scores /= num_samples
        for key in rouge_scores:
            rouge_scores[key] /= num_samples
    
    # Compile metrics
    metrics = {
        'perplexity': perplexity,
        'bleu_1': bleu_1 ,  # Convert to percentage
        'bleu_4': bleu_4 ,  # Convert to percentage
        'meteor': meteor_scores ,  # Convert to percentage
        'rouge1': rouge_scores['rouge1'],  # Convert to percentage
        'rouge2': rouge_scores['rouge2'],  # Convert to percentage
        'rougeL': rouge_scores['rougeL'],  # Convert to percentage
    }
    
    # Generate a few examples for qualitative evaluation
    examples = []
    for i in range(min(3, len(all_refs))):
        examples.append({
            'reference': all_refs[i],
            'generated': all_hyps[i]
        })
    
    # Log to wandb with a table
    if wandb.run is not None:
        examples_table = wandb.Table(columns=["Reference", "Generated"])
        for example in examples:
            examples_table.add_data(example['reference'], example['generated'])
        
        wandb.log({"generation_examples": examples_table})
    
    # Print some examples
    print("\n=== EVALUATION EXAMPLES ===")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Reference: {example['reference'][:100]}...")
        print(f"Generated: {example['generated'][:100]}...")
    
    # Print metrics
    print("\n=== EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    model.train()  
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Collect Python garbage
    return metrics, examples






