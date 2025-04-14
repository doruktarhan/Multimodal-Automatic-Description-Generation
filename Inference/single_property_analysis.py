import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom modules
from Data.custom_dataloader import CustomDataLoader
from Data.preprocessor import Preprocessor
from Model.model_utils import load_trained_model


def analyze_single_property(
    model_name: str,
    property_name: str,
    test_data_path: str,
    trained_model_path: Optional[str] = None,
    output_dir: str = "analysis_output",
    max_length: int = 1000,
    num_words_to_analyze: int = 10,
    device: str = "cuda",
    save_plots: bool = True,
):
    """
    Analyze generation for a single property with detailed token probability visualization.
    
    Args:
        model_name: Base model name
        property_name: Name of the property to analyze
        test_data_path: Path to the test data
        trained_model_path: Path to the trained model (None for base model)
        output_dir: Directory to save output files
        max_length: Maximum generation length
        num_words_to_analyze: Number of initial words to analyze token probabilities for
        device: Device to run inference on ("cuda" or "cpu")
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing property: {property_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    loader = CustomDataLoader(test_data_path)
    test_data = loader.load_all()
    print(f"Loaded {len(test_data)} test samples")
    
    # Find the property in the test data
    target_property = None
    for item in test_data:
        if item.get("property_name") == property_name:
            target_property = item
            break
    
    if target_property is None:
        print(f"Property '{property_name}' not found in test data!")
        available_properties = [item.get("property_name", "unknown") for item in test_data[:10]]
        print(f"Available properties (first 10): {available_properties}")
        return None
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_trained_model(
        base_model_name=model_name,
        trained_model_path=trained_model_path,
        quantization_bits=0,
        device_map=device
    )
    
    model_type = "base model" if trained_model_path is None else f"fine-tuned model from {trained_model_path}"
    print(f"Using {model_type}")
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Create preprocessor
    preprocessor = Preprocessor()
    
    # Generate prompt
    prompt = preprocessor.generate_prompt(target_property)
    
    # Print the prompt
    print("\n" + "="*50)
    print("PROMPT:")
    print("="*50)
    print(prompt)
    print("="*50 + "\n")
    
    # Tokenize the prompt
    input_encodings = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Decode the tokenized input to see exactly what the model sees
    actual_input = tokenizer.decode(input_encodings["input_ids"][0], skip_special_tokens=False)
    # Print the actual input that will be fed to the model
    print("\n" + "="*50)
    print("ACTUAL TOKENIZED INPUT (DECODED):")
    print("="*50)
    print(actual_input)
    print("="*50 + "\n")



    # Print tokenization info
    input_length = input_encodings["input_ids"].shape[1]
    print(f"Input length: {input_length} tokens")
    
    # Analyzing token distribution
    input_ids = input_encodings["input_ids"][0].tolist()
    input_tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
    
    print("\nToken breakdown (first 20 tokens):")
    for i, (token_id, token_text) in enumerate(zip(input_ids[:20], input_tokens[:20])):
        print(f"Token {i+1}: ID={token_id}, Text='{token_text}'")
    
    # Generate with return_dict_in_generate=True to get token probabilities
    with torch.no_grad():
        outputs = model.generate(
            **input_encodings,
            max_new_tokens=max_length,
            do_sample=False,  # Use greedy decoding for deterministic output
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
            output_hidden_states=True,
        )
    
    # Extract generated tokens and scores
    generated_ids = outputs.sequences[0][input_length:].tolist()
    token_scores = outputs.scores  # List of tensors, one per generation step
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Print the generated text
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50 + "\n")
    
    # Tokenize the generated text for analysis
    generated_tokens = [tokenizer.decode([token_id]) for token_id in generated_ids]
    
    # Analyze the first N tokens and their probabilities
    num_tokens_to_analyze = min(num_words_to_analyze * 3, len(token_scores))  # Roughly estimate tokens per word
    
    token_analysis = []
    
    for i in range(min(num_tokens_to_analyze, len(token_scores))):
        # Get the logits for the current position
        logits = token_scores[i][0]
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top 5 candidates
        topk_values, topk_indices = torch.topk(probs, 5)
        
        # Convert to list
        topk_values = topk_values.cpu().numpy().tolist()
        topk_indices = topk_indices.cpu().numpy().tolist()
        
        # Get tokens for the top 5 candidates
        topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices]
        
        # Get the chosen token
        chosen_token = generated_tokens[i] if i < len(generated_tokens) else ""
        chosen_token_id = generated_ids[i] if i < len(generated_ids) else -1
        chosen_prob = probs[chosen_token_id].item() if i < len(generated_ids) else 0
        
        token_analysis.append({
            "position": i,
            "chosen_token": chosen_token,
            "chosen_token_id": chosen_token_id,
            "chosen_probability": chosen_prob,
            "top5_tokens": topk_tokens,
            "top5_probabilities": topk_values
        })
    
    # Print token analysis
    print("\nToken probability analysis for first few tokens:")
    for i, analysis in enumerate(token_analysis):
        print(f"\nToken {i+1}: '{analysis['chosen_token']}' (ID: {analysis['chosen_token_id']}) - Probability: {analysis['chosen_probability']:.4f}")
        print("Top 5 candidates:")
        for token, prob in zip(analysis['top5_tokens'], analysis['top5_probabilities']):
            print(f"  '{token}': {prob:.4f}")
    
    # Create visualization of token probabilities
    if save_plots:
        plt.figure(figsize=(15, 8))
        
        # Plot the probabilities of selected tokens
        chosen_probs = [item["chosen_probability"] for item in token_analysis]
        positions = list(range(1, len(chosen_probs) + 1))
        
        plt.bar(positions, chosen_probs, color='blue', alpha=0.7)
        plt.xlabel('Token Position')
        plt.ylabel('Probability')
        plt.title(f'Token Probabilities for First {len(token_analysis)} Tokens')
        
        # Add token text as labels
        token_texts = [item["chosen_token"].replace('\n', '\\n') for item in token_analysis]
        plt.xticks(positions, token_texts, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{property_name.replace(' ', '_')}_token_probs.png"))
        #print(f"Saved token probability plot to {os.path.join(output_dir, f'{property_name.replace(' ', '_')}_token_probs.png')}")
        
        # Plot top-5 candidates for first few tokens
        num_tokens_viz = min(5, len(token_analysis))  # Show top-5 for first 5 tokens
        fig, axs = plt.subplots(num_tokens_viz, 1, figsize=(10, num_tokens_viz * 3))
        
        for i in range(num_tokens_viz):
            item = token_analysis[i]
            labels = [t.replace('\n', '\\n') for t in item["top5_tokens"]]
            probs = item["top5_probabilities"]
            
            ax = axs[i] if num_tokens_viz > 1 else axs
            ax.barh(labels, probs, color='green', alpha=0.7)
            ax.set_xlim(0, 1)  # Probabilities are from 0 to 1
            ax.set_title(f'Token {i+1}: Top 5 Candidates')
            
            # Highlight the chosen token
            for j, label in enumerate(labels):
                if label == item["chosen_token"]:
                    ax.get_children()[j].set_color('red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{property_name.replace(' ', '_')}_top5_viz.png"))
        #print(f"Saved top-5 candidates plot to {os.path.join(output_dir, f'{property_name.replace(' ', '_')}_top5_viz.png')}")
    
    # Save the analysis results
    analysis_results = {
        "property_name": property_name,
        "model_type": model_type,
        "prompt": prompt,
        "generated_text": generated_text,
        "token_analysis": token_analysis
    }
    
    with open(os.path.join(output_dir, f"{property_name.replace(' ', '_')}_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    #print(f"Analysis completed and saved to {os.path.join(output_dir, f'{property_name.replace(' ', '_')}_analysis.json')}")
    
    return analysis_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze generation for a single property")
    parser.add_argument("--property_name", type=str, required=True,
                        help="Name of the property to analyze")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to trained model (if None, uses base model)")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                        help="Directory to save output files")
    parser.add_argument("--max_length", type=int, default=1000,
                        help="Maximum generation length")
    parser.add_argument("--num_words", type=int, default=10,
                        help="Number of initial words to analyze token probabilities for")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable saving plots to files")
    
    args = parser.parse_args()
    
    # Get path to trained model if specified
    trained_model_path = None
    if args.trained_model_path:
        if args.trained_model_path.startswith("/"):
            # Absolute path
            trained_model_path = args.trained_model_path
        else:
            # Relative path, resolve to absolute
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            trained_model_path = os.path.join(project_dir, args.trained_model_path)
    
    analyze_single_property(
        model_name=args.base_model,
        property_name=args.property_name,
        test_data_path=args.test_data,
        trained_model_path=trained_model_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_words_to_analyze=args.num_words,
        save_plots=not args.no_plots
    )