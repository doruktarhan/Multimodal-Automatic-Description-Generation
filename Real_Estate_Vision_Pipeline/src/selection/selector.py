import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

class ImageSelector:
    def __init__(self, config: dict, selection_captions: dict):
        """
        Initialize the Image Selector with configuration and selection captions
        
        Args:
            config: Main configuration dictionary
            selection_captions: Dictionary of room-specific selection captions
        """
        self.config = config
        self.selection_captions = selection_captions
        self.room_types = list(selection_captions.keys())
        
        # Selection parameters
        self.alpha = config.get('selection', {}).get('alpha', 0.3)  # classification weight
        self.beta = config.get('selection', {}).get('beta', 0.7)   # selection weight
        self.aggregation = config.get('selection', {}).get('aggregation', 'avg')
        
        # Hierarchical thresholds
        self.tier2_threshold = 0.35
        self.tier3_threshold = 0.40
        self.max_images = 15
    
    def calculate_selection_scores(self, image_embeddings: np.ndarray, 
                                 text_embeddings_dict: Dict[str, np.ndarray],
                                 predictions: List[str]) -> np.ndarray:
        """
        Calculate selection scores for each image based on selection captions
        
        Args:
            image_embeddings: Image embeddings array
            text_embeddings_dict: Dictionary of text embeddings for each room type
            predictions: List of predicted room types for each image
            
        Returns:
            Array of selection scores
        """
        n_images = len(predictions)
        selection_scores = np.zeros(n_images)
        
        for i, predicted_room in enumerate(predictions):
            if predicted_room in text_embeddings_dict:
                # Get text embeddings for this room type
                text_embs = text_embeddings_dict[predicted_room]
                
                # Calculate similarities
                image_emb = image_embeddings[i:i+1]  # Keep 2D shape
                similarities = np.dot(image_emb, text_embs.T).flatten()
                
                # Aggregate
                if self.aggregation == 'avg':
                    selection_scores[i] = np.mean(similarities)
                elif self.aggregation == 'max':
                    selection_scores[i] = np.max(similarities)
            else:
                selection_scores[i] = 0.0
        
        return selection_scores
    
    def calculate_combined_scores(self, classification_scores: np.ndarray,
                                selection_scores: np.ndarray) -> np.ndarray:
        """
        Combine classification and selection scores
        
        Args:
            classification_scores: Array of classification confidence scores
            selection_scores: Array of selection quality scores
            
        Returns:
            Array of combined scores
        """
        return self.alpha * classification_scores + self.beta * selection_scores
    
    def hierarchical_selection(self, property_results: Dict, 
                             combined_scores: np.ndarray) -> Tuple[List[int], Dict]:
        """
        Apply hierarchical selection algorithm to choose best images
        
        Args:
            property_results: Classification results for the property
            combined_scores: Combined scores for each image
            
        Returns:
            Tuple of (selected_indices, selection_details)
        """
        predictions = property_results['predictions']
        confidence_scores = np.array(property_results['confidence_scores'])
        
        # Group images by room type with their scores
        room_to_scored_indices = {}
        for idx, room in enumerate(predictions):
            if room not in room_to_scored_indices:
                room_to_scored_indices[room] = []
            room_to_scored_indices[room].append({
                'index': idx,
                'combined_score': combined_scores[idx],
                'confidence': confidence_scores[idx]
            })
        
        # Sort images within each room by combined score (descending)
        for room in room_to_scored_indices:
            room_to_scored_indices[room].sort(
                key=lambda x: x['combined_score'], reverse=True
            )
        
        selected_indices = []
        selection_details = {
            'tier1': {},
            'tier2': {},
            'tier3': {},
            'excluded': {},
            'score_stats': {}
        }
        
        # TIER 1: Core Essential Rooms
        # Living Room - top 2
        living_room_selected = 0
        if 'living_room' in room_to_scored_indices:
            for i in range(min(2, len(room_to_scored_indices['living_room']))):
                idx_info = room_to_scored_indices['living_room'][i]
                selected_indices.append(idx_info['index'])
                living_room_selected += 1
            selection_details['tier1']['living_room'] = living_room_selected
        
        # Bedroom - top 2
        if 'bedroom' in room_to_scored_indices:
            bedroom_selected = 0
            for i in range(min(2, len(room_to_scored_indices['bedroom']))):
                idx_info = room_to_scored_indices['bedroom'][i]
                selected_indices.append(idx_info['index'])
                bedroom_selected += 1
            selection_details['tier1']['bedroom'] = bedroom_selected
        
        # Kitchen - top 2
        if 'kitchen' in room_to_scored_indices:
            kitchen_selected = 0
            for i in range(min(2, len(room_to_scored_indices['kitchen']))):
                idx_info = room_to_scored_indices['kitchen'][i]
                selected_indices.append(idx_info['index'])
                kitchen_selected += 1
            selection_details['tier1']['kitchen'] = kitchen_selected
        
        # Bathroom - top 2
        if 'bathroom' in room_to_scored_indices:
            bathroom_selected = 0
            for i in range(min(2, len(room_to_scored_indices['bathroom']))):
                idx_info = room_to_scored_indices['bathroom'][i]
                selected_indices.append(idx_info['index'])
                bathroom_selected += 1
            selection_details['tier1']['bathroom'] = bathroom_selected
        
        # Dining Room - conditional selection based on living room
        if 'dining_room' in room_to_scored_indices:
            dining_selected = 0
            if living_room_selected == 0:
                # No living room selected, take top 2 dining room
                for i in range(min(2, len(room_to_scored_indices['dining_room']))):
                    idx_info = room_to_scored_indices['dining_room'][i]
                    selected_indices.append(idx_info['index'])
                    dining_selected += 1
            else:
                # Living room selected, take only top 1 dining room
                if len(room_to_scored_indices['dining_room']) > 0:
                    idx_info = room_to_scored_indices['dining_room'][0]
                    selected_indices.append(idx_info['index'])
                    dining_selected = 1
            selection_details['tier1']['dining_room'] = dining_selected
        
        # TIER 2: High-Value Optional Rooms (confidence > 0.35)
        tier2_rooms = ['balcony', 'terrace', 'exterior']
        
        for room in tier2_rooms:
            if room in room_to_scored_indices and len(room_to_scored_indices[room]) > 0:
                top_image = room_to_scored_indices[room][0]
                if top_image['confidence'] > self.tier2_threshold:
                    selected_indices.append(top_image['index'])
                    selection_details['tier2'][room] = 1
                else:
                    selection_details['tier2'][room] = 0
        
        # TIER 3: Supporting Rooms (confidence > 0.40)
        tier3_rooms = ['stairway', 'hallway', 'home_office', 'laundry_room', 
                      'garage', 'basement']
        
        for room in tier3_rooms:
            if room in room_to_scored_indices and len(room_to_scored_indices[room]) > 0:
                top_image = room_to_scored_indices[room][0]
                if top_image['confidence'] > self.tier3_threshold:
                    selected_indices.append(top_image['index'])
                    selection_details['tier3'][room] = 1
                else:
                    selection_details['tier3'][room] = 0
        
        # Track excluded classes
        excluded_classes = ['floor_plan', 'outside', 'garden']
        for room in excluded_classes:
            if room in room_to_scored_indices:
                selection_details['excluded'][room] = len(room_to_scored_indices[room])
        
        # Ensure we don't exceed max_images
        if len(selected_indices) > self.max_images:
            selected_indices = selected_indices[:self.max_images]
        
        # Calculate statistics for selected images
        if selected_indices:
            selected_combined_scores = combined_scores[selected_indices]
            selected_confidence_scores = confidence_scores[selected_indices]
            
            selection_details['score_stats'] = {
                'combined': {
                    'mean': float(np.mean(selected_combined_scores)),
                    'std': float(np.std(selected_combined_scores)),
                    'min': float(np.min(selected_combined_scores)),
                    'max': float(np.max(selected_combined_scores))
                },
                'confidence': {
                    'mean': float(np.mean(selected_confidence_scores)),
                    'std': float(np.std(selected_confidence_scores)),
                    'min': float(np.min(selected_confidence_scores)),
                    'max': float(np.max(selected_confidence_scores))
                }
            }
        
        selection_details['total_selected'] = len(selected_indices)
        selection_details['available_rooms'] = list(room_to_scored_indices.keys())
        
        return selected_indices, selection_details
    
    def select_images(self, property_name: str, 
                     image_embeddings: np.ndarray,
                     text_embeddings_dict: Dict[str, np.ndarray],
                     classification_results: Dict) -> Dict:
        """
        Main method to select best images for a property
        
        Args:
            property_name: Name of the property
            image_embeddings: Image embeddings array
            text_embeddings_dict: Text embeddings for selection captions
            classification_results: Classification results for the property
            
        Returns:
            Dictionary containing selection results
        """
        # Calculate selection scores
        selection_scores = self.calculate_selection_scores(
            image_embeddings,
            text_embeddings_dict,
            classification_results['predictions']
        )
        
        # Get classification scores
        classification_scores = np.array(classification_results['confidence_scores'])
        
        # Calculate combined scores
        combined_scores = self.calculate_combined_scores(
            classification_scores,
            selection_scores
        )
        
        # Apply hierarchical selection
        selected_indices, selection_details = self.hierarchical_selection(
            classification_results,
            combined_scores
        )
        
        # Compile results
        results = {
            'property_name': property_name,
            'timestamp': datetime.now().isoformat(),
            'num_images': classification_results['num_images'],
            'num_selected': len(selected_indices),
            'selected_indices': selected_indices,
            'selected_image_paths': [
                classification_results['image_paths'][idx] 
                for idx in selected_indices
            ],
            'selected_predictions': [
                classification_results['predictions'][idx] 
                for idx in selected_indices
            ],
            'selection_scores': selection_scores.tolist(),
            'combined_scores': combined_scores.tolist(),
            'selection_details': selection_details,
            'config': {
                'alpha': self.alpha,
                'beta': self.beta,
                'aggregation': self.aggregation,
                'tier2_threshold': self.tier2_threshold,
                'tier3_threshold': self.tier3_threshold,
                'max_images': self.max_images
            }
        }
        
        return results