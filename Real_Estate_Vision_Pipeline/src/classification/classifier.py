import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

class RoomClassifier:
    def __init__(self, config: dict, captions_config: dict):
        self.config = config
        self.room_captions = captions_config['room_captions']
        self.room_types = list(self.room_captions.keys())
        self.aggregation = config['classification']['aggregation']
        
        # Create a unique identifier for this caption configuration
        # This helps identify when captions have changed
        caption_str = json.dumps(self.room_captions, sort_keys=True)
        self.caption_hash = hashlib.md5(caption_str.encode()).hexdigest()[:8]
    
    def get_text_embeddings(self, extractor, manager) -> Dict[str, np.ndarray]:
        """
        Get or compute text embeddings for current run
        
        This method:
        1. Creates a unique identifier for the current run
        2. Checks if text embeddings already exist for this run
        3. If not, extracts them and saves for reuse within the run
        """
        # Create a run-specific identifier
        caption_style = self.config['classification']['caption_style']
        run_id = f"classification_{caption_style}_{self.caption_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Try to load cached embeddings for this run
        cached = manager.load_text_embeddings(run_id)
        
        if cached is not None:
            print("Using cached text embeddings for this run")
            return cached
        
        print("Computing new text embeddings...")
        print(f"Run ID: {run_id}")
        text_embeddings_dict = {}
        
        # Extract embeddings for each room type
        for room, captions in self.room_captions.items():
            print(f"  Processing {room}: {len(captions)} captions")
            embeddings = extractor.extract_text_embeddings(captions)
            text_embeddings_dict[room] = embeddings
        
        # Save for reuse within this run
        manager.save_text_embeddings(run_id, text_embeddings_dict)
        
        # Store run_id for potential cleanup later
        self.current_run_id = run_id
        
        return text_embeddings_dict
    
    def classify_property(self, image_embeddings: np.ndarray, 
                         text_embeddings_dict: Dict[str, np.ndarray],
                         image_paths: List[str]) -> Dict:
        """
        Classify all images in a property
        
        Returns comprehensive classification results including:
        - Predictions for each image
        - Full similarity matrix
        - Confidence scores
        - Top-k predictions
        """
        # Calculate similarities
        similarity_scores = self._calculate_similarities(image_embeddings, text_embeddings_dict)
        
        # Get predictions and confidence scores
        predictions = []
        confidence_scores = []
        top_3_predictions = []
        
        for scores in similarity_scores:
            # Get prediction (highest scoring room)
            pred_idx = np.argmax(scores)
            predictions.append(self.room_types[pred_idx])
            confidence_scores.append(float(scores[pred_idx]))
            
            # Get top 3 predictions
            top_3_idx = np.argsort(scores)[-3:][::-1]
            top_3 = [(self.room_types[idx], float(scores[idx])) for idx in top_3_idx]
            top_3_predictions.append(top_3)
        
        # Calculate room distribution
        room_distribution = dict(Counter(predictions))
        
        # Ensure all room types are in distribution (even with 0 count)
        for room in self.room_types:
            if room not in room_distribution:
                room_distribution[room] = 0
        
        return {
            'predictions': predictions,
            'similarity_matrix': similarity_scores.tolist(),
            'confidence_scores': confidence_scores,
            'top_3_predictions': top_3_predictions,
            'room_distribution': room_distribution,
            'image_paths': image_paths,
            'num_images': len(image_paths),
            'room_types': self.room_types,
            'confidence_stats': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            }
        }
    
    def _calculate_similarities(self, image_embeddings: np.ndarray, 
                               text_embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate similarity scores with specified aggregation method
        """
        n_images = image_embeddings.shape[0]
        n_rooms = len(self.room_types)
        similarity_scores = np.zeros((n_images, n_rooms))
        
        for i, room in enumerate(self.room_types):
            room_embeddings = text_embeddings_dict[room]
            
            # Calculate cosine similarity (dot product of normalized embeddings)
            similarities = np.dot(image_embeddings, room_embeddings.T)
            
            # Aggregate multiple caption similarities
            if self.aggregation == 'max':
                similarity_scores[:, i] = np.max(similarities, axis=1)
            else:  # 'avg'
                similarity_scores[:, i] = np.mean(similarities, axis=1)
        
        return similarity_scores
    
    def get_classification_summary(self, all_results: Dict[str, Dict]) -> Dict:
        """
        Generate summary statistics across all classified properties
        """
        total_images = 0
        all_predictions = []
        all_confidences = []
        room_totals = Counter()
        
        for property_name, results in all_results.items():
            total_images += results['num_images']
            all_predictions.extend(results['predictions'])
            all_confidences.extend(results['confidence_scores'])
            
            for room, count in results['room_distribution'].items():
                room_totals[room] += count
        
        # Calculate overall statistics
        overall_distribution = dict(room_totals)
        for room in self.room_types:
            if room not in overall_distribution:
                overall_distribution[room] = 0
        
        return {
            'total_properties': len(all_results),
            'total_images': total_images,
            'overall_room_distribution': overall_distribution,
            'confidence_statistics': {
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences))
            },
            'images_per_property': {
                'mean': float(total_images / len(all_results)),
                'min': min(r['num_images'] for r in all_results.values()),
                'max': max(r['num_images'] for r in all_results.values())
            }
        }