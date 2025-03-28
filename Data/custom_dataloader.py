#dataloader.py
import json
import os
from typing import List, Dict, Any, Iterator

class CustomDataLoader:
    """Handles loading data from various sources batch by batch in ord
    to simulate streaming for large datasets"""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        Args:
            data_path: Path to the data file or directory
        """
        self.data_path = data_path
    
    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all data at once (for small datasets)
        Returns:
            List of data items
        """
        if self.data_path.endswith('.json'):
            return self._load_json_file(self.data_path)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
    
    def stream_data(self, batch_size: int = 10) -> Iterator[List[Dict[str, Any]]]:
        """
        Stream data in batches (for larger datasets)
        
        Args:
            batch_size: Number of items to yield in each batch
            
        Yields:
            Batches of data items
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No file found at {self.data_path}")
            
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                try:
                    # Load the whole JSON array (for normal JSON files)
                    data = json.load(f)
                    
                    if not isinstance(data, list):
                        raise ValueError(f"Expected JSON array but got {type(data)}")
                    
                    # Process in batches to simulate streaming
                    for i in range(0, len(data), batch_size):
                        yield data[i:min(i + batch_size, len(data))]
                        
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing JSON: {e}")
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
        
    def _load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file
        Args:
            file_path: Path to the JSON file   
        Returns:
            List of data items from the JSON file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)