import os
import sys
import torch
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm

# Add path to find modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from Model.model_utils import load_tokenizer, load_model
from Data.preprocessor import Preprocessor
from Data.new_dataset_loader_function import RealEstateLoader  # Assuming you've saved the previous file as realestate_loader.py

def train():
    """Simple training script for real estate description generation model"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configuration
    data_path = "Data/funda_scrapped_amsterdam_sample.json"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = "outputs"
    
    # Training parameters
    batch_size = 2
    num_epochs = 3
    learning_rate = 2e-5
    warmup_steps = 100
    save_steps = 500
    gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps
    max_input_length = 4096
    max_output_length = 4096
    use_lora = True
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = load_tokenizer(model_name)
    
    # Initialize model
    logger.info(f"Loading model {model_name}...")
    model = load_model(
        model_name=model_name,
        use_lora=use_lora,
        quantization_bits=0 # Using 4-bit quantization for efficiency
    )
    
    # Initialize preprocessor with high limits to avoid truncation
    preprocessor = Preprocessor(max_input_length=32768, max_output_length=32768)
    
    # Initialize data loader
    logger.info(f"Loading data from {data_path}...")
    data_loader = RealEstateLoader(
        data_path=data_path,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_output_length=max_output_length
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps
    try:
        # First, count how many batches we have
        # This is a dummy traversal just to count batches
        num_batches = 0
        dummy_loader = RealEstateLoader(
            data_path=data_path,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            batch_size=batch_size
        )
        for _ in dummy_loader:
            num_batches += 1
        
        total_steps = num_epochs * num_batches
        logger.info(f"Total training steps: {total_steps} ({num_epochs} epochs × {num_batches} batches)")
    except Exception as e:
        logger.warning(f"Could not count total batches: {e}")
        total_steps = 1000  # Fallback value
    
    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_iterator = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        model.train()
        
        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Log loss
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_iterator.set_postfix({"loss": epoch_loss / (step + 1)})
            
            # Backward pass
            loss.backward()
            
            # Update weights with gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(epoch_iterator) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")


def evaluate(model, data_loader):
    """Dummy evaluation function placeholder"""
    # This is a placeholder for your evaluation logic
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
    
    return eval_loss


if __name__ == "__main__":
    train()