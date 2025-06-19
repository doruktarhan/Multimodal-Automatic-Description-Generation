import json
import yaml
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
from typing import List, Dict, Any, Tuple

class VisualCueExtractor:
    def __init__(self, config_file: str = "Visual_Cue_Extraction_Pipeline/config.yaml"):
        """Initialize the visual cue extraction pipeline"""
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        print("Loading VLM model...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["model"]["model_path"],
            **model_kwargs
        )
        
        self.processor = AutoProcessor.from_pretrained(self.config["model"]["model_path"])
        print("Model loaded successfully!")
        
        # Initialize error tracking
        self.errors = []
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_selection_results(self) -> Dict[str, Any]:
        """Load image selection results"""
        selection_file = self.config["data"]["selection_results"]
        with open(selection_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_property_metadata(self) -> List[Dict[str, Any]]:
        """Load property metadata as list"""
        metadata_file = self.config["data"]["property_metadata"]
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure it's a list
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    
    def load_test_metadata(self) -> List[str]:
        """Load test metadata and extract original property names (remove underscores)"""
        test_file = self.config["data"]["test_metadata"]
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Extract property names from all bins and convert back to original format
        property_names = []
        for bin_name, bin_data in test_data['bins'].items():
            for prop in bin_data['properties']:
                # Convert back to original name format (remove underscores)
                original_name = prop['name'].replace('_', ' ')
                property_names.append(original_name)
        
        return property_names
    
    def convert_name_for_selection(self, original_name: str) -> str:
        """Convert original property name to selection format (spaces to underscores)"""
        return original_name.replace(' ', '_')
    
    def get_property_info(self, original_name: str, metadata_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Get neighborhood and features for a property using original name"""
        # Find property in metadata using original name
        for prop_data in metadata_list:
            if prop_data.get('property_name') == original_name:
                neighborhood = prop_data.get('neighborhood', 'N/A')
                features = prop_data.get('features', {})
                return neighborhood, features
        
        # If not found, return defaults
        return 'N/A', {}
    
    def load_images_from_paths(self, image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
        """Load images from file paths, track missing images"""
        images = []
        missing_paths = []
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                except Exception as e:
                    missing_paths.append(path)
                    print(f"Error loading image {path}: {e}")
            else:
                missing_paths.append(path)
                print(f"Image not found: {path}")
        
        return images, missing_paths
    
    def generate_vlm_response(self, property_name: str, neighborhood: str, 
                            features: Dict[str, Any], images: List[Image.Image]) -> str:
        """Generate VLM response for a property"""
        if not images:
            return "No valid images available for analysis"
        
        # Prepare chat template
        features_json = json.dumps(features, indent=2)
        user_prompt = self.config["prompts"]["user_prompt_template"].format(
            property_name=property_name,
            neighborhood=neighborhood,
            features_json=features_json
        )
        
        # Prepare content for VLM
        content = []
        
        # Add images to content
        for img in images:
            content.append({
                "type": "image",
                "image": img
            })
        
        # Add text content
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": self.config["prompts"]["system_prompt"]
            },
            {
                "role": "user",
                "content": content,
            }
        ]
        
        # Process with VLM
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=4000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            error_msg = f"VLM processing failed: {str(e)}"
            print(f"Error processing {property_name}: {error_msg}")
            return error_msg
    
    def process_property(self, original_name: str, selection_name: str, selection_data: Dict[str, Any], 
                        property_metadata: List[Dict[str, Any]]) -> str:
        """Process a single property using original name for metadata and selection name for images"""
        print(f"Processing: {original_name}")
        
        # Get selected image paths
        selected_paths = selection_data.get("selected_image_paths", [])
        if not selected_paths:
            self.errors.append({"property": original_name, "error": "No selected_image_paths found"})
            return "No image paths available"
        
        # Load images
        images, missing_paths = self.load_images_from_paths(selected_paths)
        
        # Track missing images
        for missing_path in missing_paths:
            self.errors.append({"property": original_name, "missing_image": missing_path})
        
        if not images:
            return "No valid images could be loaded"
        
        # Get property metadata using original name
        neighborhood, features = self.get_property_info(original_name, property_metadata)
        
        # Generate VLM response
        response = self.generate_vlm_response(original_name, neighborhood, features, images)
        
        print(f"✓ Completed: {original_name} ({len(images)} images processed)")
        return response
    
    def run_extraction(self) -> Dict[str, str]:
        """Run the complete visual cue extraction pipeline"""
        print("="*60)
        print("VISUAL CUE EXTRACTION PIPELINE")
        print("="*60)
        
        # Load data
        print("Loading data files...")
        selection_results = self.load_selection_results()
        property_metadata = self.load_property_metadata()
        
        print(f"Loaded {len(property_metadata)} properties from metadata")
        print(f"Loaded {len(selection_results)} properties from selection results")
        
        # Determine which properties to process
        if self.config["test_mode"]:
            print("Running in TEST MODE")
            test_properties = self.load_test_metadata()  # These are original names now
            print(f"Test properties loaded: {len(test_properties)}")
            
            # Filter test properties that exist in original metadata and have selections
            properties_to_process = []
            for original_name in test_properties:
                # Check if exists in original metadata
                metadata_exists = any(prop.get('property_name') == original_name for prop in property_metadata)
                if not metadata_exists:
                    print(f"Warning: {original_name} not found in metadata")
                    continue
                
                # Check if selection exists (convert name for selection lookup)
                selection_name = self.convert_name_for_selection(original_name)
                if selection_name in selection_results:
                    properties_to_process.append(original_name)
                else:
                    print(f"Warning: {original_name} (as {selection_name}) not found in selection results")
            
            output_dir = self.config["results"]["test_output"]
            print(f"Test properties with both metadata and selections: {len(properties_to_process)}")
        else:
            print("Running in FULL MODE")
            # Start from original metadata, check which ones have selections
            properties_to_process = []
            for prop_data in property_metadata:
                original_name = prop_data.get('property_name')
                if not original_name:
                    continue
                
                # Check if selection exists (convert name for selection lookup)
                selection_name = self.convert_name_for_selection(original_name)
                if selection_name in selection_results:
                    properties_to_process.append(original_name)
            
            output_dir = self.config["results"]["full_output"]
        
        print(f"Total properties to process: {len(properties_to_process)}")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process properties
        results = {}
        total = len(properties_to_process)
        
        for i, original_name in enumerate(properties_to_process, 1):
            print(f"\n[{i}/{total}] Processing: {original_name}")
            
            # Convert name for selection lookup
            selection_name = self.convert_name_for_selection(original_name)
            selection_data = selection_results[selection_name]
            
            # Process using original name for results key and metadata lookup
            response = self.process_property(original_name, selection_name, selection_data, property_metadata)
            results[original_name] = response  # Save with original name
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = os.path.join(output_dir, f"visual_cues_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save errors if any
        if self.errors:
            errors_file = os.path.join(output_dir, f"errors_{timestamp}.json")
            with open(errors_file, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total properties in metadata: {len(property_metadata)}")
        print(f"Total properties in selection: {len(selection_results)}")
        print(f"Total properties processed: {len(results)}")
        print(f"Total errors logged: {len(self.errors)}")
        print(f"Results saved to: {results_file}")
        if self.errors:
            print(f"Errors saved to: {errors_file}")
        print("="*60)
        
        return results

def main():
    """Main execution function"""
    # Initialize and run extraction
    extractor = VisualCueExtractor()
    results = extractor.run_extraction()
    
    print(f"\nExtraction completed successfully!")
    print(f"Processed {len(results)} properties")

if __name__ == "__main__":
    main()