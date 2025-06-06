import json
import yaml
import os
import textwrap
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any

class PropertyAnalysisPipeline:
    def __init__(self, config_file: str = "Image_Test_QwenVL/config.yaml"):
        """
        Initialize the property analysis pipeline with Qwen 2.5 VL model
        """
        # Load configuration
        self.config = self.load_config(config_file)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["model"]["model_path"],
            **model_kwargs
        )
        
        self.processor = AutoProcessor.from_pretrained(self.config["model"]["model_path"])
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
        
    def load_property_metadata(self, property_name: str) -> Dict[str, Any]:
        """
        Load property metadata from JSON file
        """
        metadata_file = self.config["paths"]["metadata_file"]
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for property_data in data:
                if property_data.get('property_name') == property_name:
                    return property_data
        elif isinstance(data, dict):
            if data.get('property_name') == property_name:
                return data
                
        return {}
    
    def load_property_images(self, property_name: str, photo_indices: List[int], 
                           resolution: str) -> List[Image.Image]:
        """
        Load property images based on indices
        """
        images = []
        property_name_normalized = property_name.replace(" ", "_")
        base_path = self.config["paths"]["base_path"]
        property_path = Path(base_path) / property_name_normalized / resolution
        
        for idx in photo_indices:
            image_path = property_path / f"photo_{idx:02d}.jpg"
            if image_path.exists():
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                print(f"Loaded: {image_path}")
        
        return images
    
    def generate_chat_template(self, item: dict) -> dict[str, str]:
        system_prompt = self.config["prompts"]["system_prompt"]
        
        user_prompt = self.config["prompts"]["user_prompt_template"].format(
            property_name=item.get('property_name', 'N/A'),
            neighborhood=item.get('neighborhood', 'N/A'),
            features_json=json.dumps(item.get('features', {}), indent=2)
        )

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}
    
    def analyze_property(self, property_name: str = None, photo_indices: List[int] = None, 
                        resolution: str = None) -> str:
        """
        Analyze property using images and metadata
        """
        # Use config defaults if parameters not provided
        if property_name is None:
            property_name = self.config["test_settings"]["property_name"]
        if photo_indices is None:
            photo_indices = self.config["test_settings"]["photo_indices"]
        if resolution is None:
            resolution = self.config["test_settings"]["resolution"]
        
        # Load property data
        print(f"Loading metadata for property: {property_name}")
        property_data = self.load_property_metadata(property_name)
        
        # Generate chat template
        chat_template = self.generate_chat_template(property_data)
        
        # Load images
        print(f"Loading images: {photo_indices}")
        images = self.load_property_images(property_name, photo_indices, resolution)
        
        if not images:
            return "No valid images found for analysis"
        
        print(f"Successfully loaded {len(images)} images")
        
        # Prepare messages for the model
        content = []
        
        # Add images to content
        for i, img in enumerate(images):
            content.append({
                "type": "image",
                "image": img
            })
        
        # Add text content
        content.append({
            "type": "text", 
            "text": chat_template["user_prompt"]
        })
        
        messages = [
            {
                "role": "system",
                "content": chat_template["system_prompt"]
            },
            {
                "role": "user",
                "content": content,
            }
        ]
        
        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Debug: save input text
        with open(f"debug_input_{property_name.replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        
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
        print("Generating analysis...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=4000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Debug: save output text
        with open(f"debug_output_{property_name.replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
            f.write(output_text[0])
        
        return output_text[0]

def test_pipeline():
    """
    Test the property analysis pipeline
    """
    # Initialize pipeline (reads from config.yaml)
    pipeline = PropertyAnalysisPipeline()
    
    # Run analysis using config settings
    result = pipeline.analyze_property()
    
    # Or override specific parameters
    # result = pipeline.analyze_property(
    #     property_name="Different Property",
    #     photo_indices=[2, 4, 6],
    #     resolution="180x120"
    # )
    
    print("="*50)
    print("ANALYSIS RESULT:")
    print("="*50)
    print(result)
    
    return result

if __name__ == "__main__":
    test_pipeline()