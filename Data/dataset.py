#dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


class RealEstateDataset(Dataset):
    """Dataset Pytorch class for real estate description generation"""

    def __init__(
            self,
            examples: List[Dict[str,str]],
            tokenizer,
            max_input_length: int = 32768,
            max_output_length: int = 32768
            ):
        """
        Initialize the dataset with examples and tokenizer
        
        Args:
            examples: List of preprocessed examples with 'input' and 'output' keys
            tokenizer: Tokenizer for the language model
            max_input_length: Maximum length for input sequences
            max_output_length: Maximum length for output sequences
        """       
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        """Return the number of examples"""
        return len(self.examples)

    def __getitem__(self,idx):
        """
        Get a tokenized example at the given index
        
        Args:
            idx: Index of the example to get
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """

        example = self.examples[idx]
        input_text = example['input']
        output_text = example['output']

        # Tokenize input and output separately
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,  # No padding at item level
            return_tensors=None  # Return lists, not tensors
        )
        
        output_encodings = self.tokenizer(
            output_text + self.tokenizer.eos_token,
            max_length=self.max_output_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Combine input and output tokens
        input_ids = input_encodings["input_ids"] + output_encodings["input_ids"]
        attention_mask = input_encodings["attention_mask"] + output_encodings["attention_mask"]
        
        # Create labels: -100 for input tokens, actual tokens for output
        labels = [-100] * len(input_encodings["input_ids"]) + output_encodings["input_ids"]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }
    
    #instead of creating the examples and than create the dataset with those examples,
    #we can create the dataset directly from the raw data calling this method
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
            preprocessor: Preprocessor to convert raw data to examples
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
            examples: List[Dict[str,str]],
            tokenizer,
            max_input_length: int = 32768,
            ):
        """
        Initialize the dataset with examples and tokenizer
        
        Args:
            examples: List of preprocessed examples with 'input' key only
            tokenizer: Tokenizer for the language model
            max_input_length: Maximum length for input sequences
        """       
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        """Return the number of examples"""
        return len(self.examples)
        
    def __getitem__(self, idx):
        """
        Get a tokenized example at the given index
        
        Args:
            idx: Index of the example to get
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        example = self.examples[idx]
        input_text = example['input']

        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,  # No padding at item level
            return_tensors=None  # Return lists, not tensors
        )
        
        # Get input tokens
        input_ids = input_encodings["input_ids"] 
        attention_mask = input_encodings["attention_mask"]  
        
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
    ):
        """
        Create a dataset directly from raw data using a preprocessor
        
        Args:
            raw_data: List of raw property data dictionaries
            preprocessor: Preprocessor to convert raw data to examples
            tokenizer: Tokenizer for the language model
            max_input_length: Maximum length for input sequences
            
        Returns:
            Initialized RealEstateDatasetForInference
        """
        # For inference, we only need the input part
        examples = []
        for item in raw_data:
            # Generate prompt for this item
            prompt = preprocessor.generate_prompt(item)
            examples.append({"input": prompt})
        
        return cls(examples, tokenizer, max_input_length)


def collate_fn(batch):
    """
    Custom collation function for dynamic padding of batches
    
    Args:
        batch: List of dictionaries, each with input_ids, attention_mask, and labels
        
    Returns:
        Batch dictionary with padded tensors
    """
    # Determine max length in this batch
    max_input_len = max([len(x["input_ids"]) for x in batch])
    
    # Initialize padded tensors
    input_ids = []
    attention_mask = []
    labels = []
    
    # Get pad token id from the first example (safer approach)
    pad_token_id = 0  # Default pad token ID
    
    # Check if this batch contains labels (training) or not (inference)
    has_labels = "labels" in batch[0]
    
    # Pad sequences to max length in batch
    for item in batch:
        # Get current lengths
        curr_len = len(item["input_ids"])
        pad_len = max_input_len - curr_len
        
        # Skip padding if not needed
        if pad_len == 0:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            if has_labels:
                labels.append(item["labels"])
            continue
        
        # Pad input_ids with pad_token_id
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        
        # Pad attention_mask with 0s
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        
        # Pad labels with -100 if they exist
        if has_labels:
            padded_labels = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
            labels.append(padded_labels)
    
    # Stack tensors
    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }
    
    # Add labels if they exist
    if has_labels:
        result["labels"] = torch.stack(labels)
    
    return result