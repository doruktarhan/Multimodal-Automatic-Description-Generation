import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import datetime

class EmbeddingManager:
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def save_property_embeddings(self, property_name: str, embeddings: np.ndarray, 
                                image_paths: List[str], metadata: dict):
        """Save embeddings for a single property"""
        filename = self.embeddings_dir / f"{property_name}_embeddings.pkl"
        
        data = {
            'property_name': property_name,
            'embeddings': embeddings,
            'image_paths': image_paths,
            'metadata': metadata,
            'embedding_dim': embeddings.shape[1],
            'num_images': len(image_paths)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        return filename
    
    def load_property_embeddings(self, property_name: str) -> Optional[Dict]:
        """Load embeddings for a single property"""
        filename = self.embeddings_dir / f"{property_name}_embeddings.pkl"
        
        if filename.exists():
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
    
    def list_completed_properties(self) -> List[str]:
        """List all properties with saved embeddings"""
        completed = []
        for file in self.embeddings_dir.glob("*_embeddings.pkl"):
            property_name = file.stem.replace("_embeddings", "")
            completed.append(property_name)
        return completed
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings"""
        stats = {
            'total_properties': 0,
            'total_embeddings': 0,
            'embedding_dimension': None,
            'properties': []
        }
        
        for file in self.embeddings_dir.glob("*_embeddings.pkl"):
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            stats['total_properties'] += 1
            stats['total_embeddings'] += data['num_images']
            stats['embedding_dimension'] = data['embedding_dim']
            stats['properties'].append({
                'name': data['property_name'],
                'num_images': data['num_images']
            })
        
        return stats
    

    def save_text_embeddings(self, run_id: str, embeddings_dict: Dict[str, np.ndarray]):
        """
        Save text embeddings for a specific run
        
        Args:
            run_id: Unique identifier for this run
            embeddings_dict: Dictionary mapping room types to their text embeddings
        """
        # Create text embeddings directory if it doesn't exist
        text_dir = self.embeddings_dir.parent / 'text_embeddings'
        text_dir.mkdir(parents=True, exist_ok=True)
        
        filename = text_dir / f"{run_id}.pkl"
        
        data = {
            'run_id': run_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'embeddings': embeddings_dict,
            'room_types': list(embeddings_dict.keys()),
            'embedding_shapes': {room: emb.shape for room, emb in embeddings_dict.items()}
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved text embeddings to: {filename.name}")

    def load_text_embeddings(self, run_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load text embeddings for a specific run
        
        Args:
            run_id: Unique identifier for the run
            
        Returns:
            Dictionary of text embeddings or None if not found
        """
        text_dir = self.embeddings_dir.parent / 'text_embeddings'
        filename = text_dir / f"{run_id}.pkl"
        
        if filename.exists():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data['embeddings']
        return None

    def cleanup_old_text_embeddings(self, keep_last_n: int = 5):
        """
        Clean up old text embedding files, keeping only the most recent ones
        
        Args:
            keep_last_n: Number of recent files to keep
        """
        text_dir = self.embeddings_dir.parent / 'text_embeddings'
        if not text_dir.exists():
            return
        
        # Get all text embedding files
        files = list(text_dir.glob("classification_*.pkl"))
        
        if len(files) <= keep_last_n:
            return
        
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove old files
        files_to_remove = files[:-keep_last_n]
        for file in files_to_remove:
            file.unlink()
        
        if files_to_remove:
            print(f"Cleaned up {len(files_to_remove)} old text embedding files")