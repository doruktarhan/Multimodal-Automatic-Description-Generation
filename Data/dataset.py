#dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

from Data.chat_template_utils import is_qwen_text_model, get_qwen_chat_template, initialize_chat_template


class RealEstateDataset(Dataset):
    """Dataset Pytorch class for real estate description generation using chat template"""

    def __init__(
            self,
            examples: List[List[Dict]],  # List of message lists (each item is a conversation)
            tokenizer,
            max_input_length: int = 32768,
            max_output_length: int = 32768
            ):
        """
        Initialize the dataset with chat examples and tokenizer
        
        Args:
            examples: List of conversations where each conversation is a list of message dicts
                     Each message dict should have 'role' and 'content' keys
            tokenizer: Tokenizer for the language model
            max_input_length: Maximum length for input sequences
            max_output_length: Maximum length for output sequences
        """       
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Initialize the chat template
        self.template_modified = initialize_chat_template(self.tokenizer)

    def __len__(self):
        """Return the number of examples"""
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a tokenized example at the given index
        
        Args:
            idx: Index of the example to get
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        messages = self.examples[idx]

        max_length = self.max_input_length + self.max_output_length



        rendered_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            max_length=max_length,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True
        )

        input_ids = rendered_text["input_ids"]
        attention_mask = rendered_text["attention_mask"]
        assistant_mask = rendered_text["assistant_masks"]

        # Set labels to -100 for non-assistant tokens (they'll be ignored in loss computation)
        labels = torch.tensor([-100 if mask == 0 else token for token, mask in zip(input_ids, assistant_mask)])
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }
    
    @classmethod
    def from_preprocessor(
        cls, 
        raw_data: List[Dict[str, Any]], 
        preprocessor, 
        tokenizer, 
        max_input_length: int = 32768, 
        max_output_length: int = 32768
    ):
        """
        Create a dataset directly from raw data using a preprocessor
        
        Args:
            raw_data: List of raw property data dictionaries
            preprocessor: Preprocessor to convert raw data to chat examples
            tokenizer: Tokenizer for the language model
            max_input_length: Maximum length for input sequences
            max_output_length: Maximum length for output sequences
            
        Returns:
            Initialized RealEstateDataset
        """
        examples = preprocessor.process_data(raw_data)
        return cls(examples, tokenizer, max_input_length, max_output_length)
    

class RealEstateDatasetForInference(Dataset):
    """Dataset Pytorch class for real estate description inference"""
    
    def __init__(
            self,
            examples: List,  # This can be either list of message lists or list of dicts
            tokenizer,
            max_input_length: int = 32768,
            ):
        """
        Initialize the dataset with examples and tokenizer
        """       
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
        # Initialize the chat template
        self.template_modified = initialize_chat_template(self.tokenizer)

    def __len__(self):
        """Return the number of examples"""
        return len(self.examples)
        
    def __getitem__(self, idx):
        """
        Get a tokenized example at the given index
        """
        example = self.examples[idx]
        
        # Check if example is in chat format (list of message dicts) or single dict format
        if isinstance(example, list):
            # Chat format - use apply_chat_template
            messages = example
            rendered_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                max_length=self.max_input_length,
                add_generation_prompt=True,
                return_assistant_tokens_mask=True,
                return_dict=True
            )
            
            input_ids = rendered_text["input_ids"]
            attention_mask = rendered_text["attention_mask"]
        
        elif isinstance(example, dict) and "input" in example:
            # Old format with plain text input
            input_text = example["input"]
            
            # Tokenize input
            encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
        
        else:
            raise ValueError(f"Unsupported example format: {type(example)}")
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
    
    @classmethod
    def from_preprocessor(
        cls, 
        raw_data: List[Dict[str, Any]], 
        preprocessor, 
        tokenizer, 
        max_input_length: int = 32768, 
        use_chat_format: bool = True,  # New parameter to control format
    ):
        """
        Create a dataset directly from raw data using a preprocessor
        """
        examples = []
        
        if use_chat_format:
            # Create examples in chat format
            for item in raw_data:
                chat_msgs = preprocessor.generate_chat_template(item)
                messages = [
                    {"role": "system", "content": chat_msgs["system_prompt"]},
                    {"role": "user", "content": chat_msgs["user_prompt"]}
                ]
                examples.append(messages)
        else:
            # Create examples in old format with "input" key
            for item in raw_data:
                prompt = preprocessor.generate_prompt(item)
                examples.append({"input": prompt})
        
        return cls(examples, tokenizer, max_input_length)



def collate_fn(batch, pad_token_id=0):
    """
    Custom collate function for chat template-based dataset
    
    Args:
        batch: List of dictionaries with input_ids, attention_mask, and optional labels
        pad_token_id: Token ID to use for padding (default: 0)
        
    Returns:
        Dictionary with padded and batched tensors
    """
    # Extract the max length in this batch
    max_length = max([len(item['input_ids']) for item in batch])
    
    # Initialize lists for batch items
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    
    # Process each item in the batch
    for item in batch:
        # Get the current length
        curr_length = len(item['input_ids'])
        padding_length = max_length - curr_length
        
        # Pad the input_ids with pad_token_id
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        input_ids_batch.append(padded_input_ids)
        
        # Pad the attention_mask with zeros
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(padding_length, dtype=torch.long)
        ])
        attention_mask_batch.append(padded_attention_mask)
        
        # Only process labels if they exist in the item
        if 'labels' in item:
            padded_labels = torch.cat([
                item['labels'],
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            labels_batch.append(padded_labels)
    
    # Stack all tensors into batches
    result = {
        "input_ids": torch.stack(input_ids_batch),
        "attention_mask": torch.stack(attention_mask_batch),
    }
    
    # Only include labels in the result if they were present
    if labels_batch:
        result["labels"] = torch.stack(labels_batch)
    
    return result