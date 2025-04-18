import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from typing import Dict, Any, Optional, Tuple

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


def create_quantization_config(bits: int = 4, use_double_quant: bool = True):
    """
    Create a configuration for model quantization using BitsAndBytesConfig.
    
    Args:
        bits: Number of bits for quantization (4 or 8)
        use_double_quant: Whether to use double quantization for further memory saving
        
    Returns:
        BitsAndBytesConfig object
    """
    if bits not in [4, 8]:
        raise ValueError(f"Bits must be 4 or 8, got {bits}")
    
    # Use the official BitsAndBytesConfig class instead of raw parameters
    return BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type="nf4" if bits == 4 else "fp4"
    )


def load_model(
    model_name: str,
    use_lora: bool = True,
    lora_config: Optional[LoraConfig] = None,
    quantization_bits: int = 8,
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
        model_kwargs["quantization_config"] = quant_config  # Use the new parameter name
    
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

def load_trained_model(
        base_model_name:str,
        trained_model_path: Optional[str] = None,
        quantization_bits:int = 8,
        device_map:str = "auto",
        )-> Tuple[torch.nn.Module,AutoTokenizer]:
    
    base_model = load_model(
        base_model_name,
        use_lora= False,
        quantization_bits=quantization_bits,
        device_map=device_map
        )
    
    tokenizer = load_tokenizer(base_model_name)

    if trained_model_path is not None:
        # Load the trained model
        print(f"Loading trained model from {trained_model_path}...")
        trained_model = PeftModel.from_pretrained(
            base_model,
            trained_model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            local_files_only = True #ensures to look for local files
        )
        print(f"Trained model loaded from {trained_model_path}")
    else: 
        trained_model = base_model
        # Load the base model without LoRA
        print(f"Base model loaded from {base_model_name}")
    
        # Enable KV caching for faster inference
    if hasattr(trained_model, "config"):
        trained_model.config.use_cache = True
    
    return trained_model, tokenizer

# def load_trained_model(
#         base_model_name: str,
#         trained_model_path: Optional[str] = None,
#         quantization_bits: int = 4,
#         device_map: str = "auto",
#         ) -> Tuple[torch.nn.Module, AutoTokenizer]:
#     """
#     Load a base model and optionally apply LoRA weights
    
#     Args:
#         base_model_name: Name or path of the base model
#         trained_model_path: Path to trained LoRA weights (can be local or HF Hub)
#         quantization_bits: Quantization bits (0, 4, or 8)
#         device_map: Device mapping strategy
        
#     Returns:
#         Tuple of (model, tokenizer)
#     """
#     base_model = load_model(
#         base_model_name,
#         use_lora=False,
#         quantization_bits=quantization_bits,
#         device_map=device_map
#         )
    
#     tokenizer = load_tokenizer(base_model_name)

#     if trained_model_path is not None:
#         # Check if this is a local path or HF model ID
#         is_local_path = os.path.exists(trained_model_path)
        
#         if is_local_path:
#             print(f"Loading trained model from local path: {trained_model_path}")
            
#             # Check which structure we have (direct or with adapter_model subdir)
#             adapter_config_path = os.path.join(trained_model_path, "adapter_config.json")
#             adapter_model_subdir = os.path.join(trained_model_path, "adapter_model")
            
#             # If adapter_config is in a subdirectory, use that path
#             if not os.path.exists(adapter_config_path) and os.path.exists(adapter_model_subdir):
#                 adapter_subdir_config = os.path.join(adapter_model_subdir, "adapter_config.json")
#                 if os.path.exists(adapter_subdir_config):
#                     print(f"Found adapter config in subdirectory")
#                     trained_model_path = adapter_model_subdir
            
#             try:
#                 trained_model = PeftModel.from_pretrained(
#                     base_model,
#                     trained_model_path,
#                     device_map=device_map,
#                     torch_dtype=torch.float16,
#                     local_files_only=True  # Important: Only use local files
#                 )
#                 print(f"Trained model loaded from {trained_model_path}")
#             except Exception as e:
#                 print(f"Error loading model: {str(e)}")
#                 print("Checking for alternative adapter paths...")
                
#                 # Try with different common structures
#                 possible_paths = [
#                     os.path.join(trained_model_path, "adapter_model"),
#                     os.path.dirname(trained_model_path),  # Try parent directory
#                     os.path.join(os.path.dirname(trained_model_path), "adapter_model")
#                 ]
                
#                 for path in possible_paths:
#                     if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
#                         print(f"Trying alternative path: {path}")
#                         try:
#                             trained_model = PeftModel.from_pretrained(
#                                 base_model,
#                                 path,
#                                 device_map=device_map,
#                                 torch_dtype=torch.float16,
#                                 local_files_only=True
#                             )
#                             print(f"Successfully loaded model from {path}")
#                             break
#                         except Exception as nested_error:
#                             print(f"Failed with path {path}: {str(nested_error)}")
#                 else:
#                     # If all paths failed, just use the base model
#                     print("Could not load trained model, using base model instead")
#                     trained_model = base_model
#         else:
#             # Assume it's a Hugging Face Hub model ID
#             print(f"Loading trained model from Hugging Face Hub: {trained_model_path}")
#             try:
#                 trained_model = PeftModel.from_pretrained(
#                     base_model,
#                     trained_model_path,
#                     device_map=device_map,
#                     torch_dtype=torch.float16
#                 )
#                 print(f"Trained model loaded from HF Hub: {trained_model_path}")
#             except Exception as e:
#                 print(f"Error loading model from HF Hub: {str(e)}")
#                 print("Using base model instead")
#                 trained_model = base_model
#     else: 
#         trained_model = base_model
#         # Load the base model without LoRA
#         print(f"Base model loaded from {base_model_name}")
    
#     # Enable KV caching for faster inference
#     if hasattr(trained_model, "config"):
#         trained_model.config.use_cache = True
    
#     return trained_model, tokenizer  

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


