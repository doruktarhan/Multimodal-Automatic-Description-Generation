import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import Dict, Any, Optional

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the tokenizer for a specific model.
    
    Args:
        model_name: Name or path of the pretrained model
        
    Returns:
        The loaded tokenizer
    """
    print(f"Loading tokenizer for {model_name}...")  

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #trust_remote_code=True  #required for some models, not in Qwen 
    )  

    if tokenizer.pad_token is None:
        # For models without pad token, use EOS token instead
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer   



def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Create a LoRA configuration for efficient fine-tuning.
    
    Args:
        r: Rank of the update matrices
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to (if None, uses defaults)
        
    Returns:
        LoRA configuration
    """
    # Default target modules for transformer-based models
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj"      # FFN modules
        ]
    
    # Create and return the LoRA configuration
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )


def create_quantization_config(bits: int = 4, use_double_quant: bool = True) -> Dict[str, Any]:
    """
    Create a configuration for model quantization.
    
    Args:
        bits: Number of bits for quantization (4 or 8)
        use_double_quant: Whether to use double quantization for further memory saving
        
    Returns:
        Quantization configuration dictionary
    """
    if bits not in [4, 8]:
        raise ValueError(f"Bits must be 4 or 8, got {bits}")
    
    # Configure quantization (bitsandbytes library)
    return {
        f"load_in_{bits}bit": True,
        f"bnb_{bits}bit_quant_type": "nf4" if bits == 4 else "fp8",
        f"bnb_{bits}bit_compute_dtype": torch.float16,
        f"bnb_{bits}bit_use_double_quant": use_double_quant
    }


def load_model(
    model_name: str,
    use_lora: bool = True,
    lora_config: Optional[LoraConfig] = None,
    quantization_bits: int = 4,
    device_map: str = "auto"
) -> torch.nn.Module:
    """
    Load and prepare a model for fine-tuning.
    
    Args:
        model_name: Name or path of the pretrained model
        use_lora: Whether to apply LoRA for parameter-efficient fine-tuning
        lora_config: LoRA configuration (if None, creates with default params)
        quantization_bits: Number of bits for quantization (0 = disable quantization)
        device_map: How to map model layers to devices ('auto' or specific mapping)
        
    Returns:
        The loaded and prepared model
    """
    print(f"Loading model {model_name}...")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch.float16
    }
    
    # Add quantization config if requested
    if quantization_bits in [4, 8]:
        quant_config = create_quantization_config(quantization_bits)
        model_kwargs.update(quant_config)
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Apply LoRA if requested
    if use_lora:
        # Create default LoRA config if not provided
        if lora_config is None:
            lora_config = create_lora_config()
        
        # Prepare model for k-bit training if using quantization
        if quantization_bits in [4, 8]:
            model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters to the model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        model.print_trainable_parameters()
    
    return model

def save_model(model, output_dir: str):
    """
    Save a model's weights.
    
    Args:
        model: The model to save
        output_dir: Directory to save the model to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    if hasattr(model, "save_pretrained"):
        # For models with save_pretrained method (Hugging Face models)
        model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    else:
        # Fallback for regular PyTorch models
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        print(f"Model state dict saved to {os.path.join(output_dir, 'model.pt')}")


