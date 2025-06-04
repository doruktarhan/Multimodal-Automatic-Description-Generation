import json
import os
import textwrap
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any

class PropertyAnalysisPipeline:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the property analysis pipeline with Qwen 2.5 VL model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor with conditional flash attention
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        # Only add flash_attention_2 if CUDA is available (not on Mac)
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
    def load_property_metadata(self, property_name: str, metadata_file: str = "property_data.json") -> Dict[str, Any]:
        """
        Load property metadata from JSON file
        """
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
                           resolution: str = "360x240", base_path: str = "./properties") -> List[Image.Image]:
        """
        Load property images based on indices
        """
        images = []
        property_name_normalized = property_name.replace(" ", "_")
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
        system_prompt = (
            "You are an expert real estate copywriter located in Amsterdam. Your task is to create a single, "
            "compelling, and informative property description based on the provided metadata. "
            "Your main goal is to engage potential buyers or renters and highlight what makes the property attractive.\n\n"
            "Use the metadata as your factual basis. You have creative freedom in how you structure the description, "
            "the language you use, and the aspects you choose to emphasize to best showcase the property and its "
            "location (if neighborhood information is provided in the metadata).\n\n"
            "Aim for a professional, positive, and persuasive tone. Ensure the core factual details from the metadata "
            "(like number of rooms, essential features) are naturally woven into your narrative. Avoid inventing "
            "specific measurements or highly unique amenities if they are not mentioned in the metadata."
        )
        
        raw_user = f"""
        Property name: {item.get('property_name', 'N/A')}
        Neighbourhood: {item.get('neighborhood', 'N/A')}

        FEATURES JSON:
        {json.dumps(item.get('features', {}), indent=2)}

        Write a compelling property description that highlights the main features and benefits below.
        """

        user_prompt = textwrap.dedent(raw_user).strip()

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}
    
    def analyze_property(self, property_name: str, photo_indices: List[int], 
                        resolution: str = "360x240", base_path: str = "./properties", 
                        metadata_file: str = "property_data.json") -> str:
        """
        Analyze property using images and metadata
        """
        # Load property data
        print(f"Loading metadata for property: {property_name}")
        property_data = self.load_property_metadata(property_name, metadata_file)
        
        # Generate chat template
        chat_template = self.generate_chat_template(property_data)
        
        # Load images
        print(f"Loading images: {photo_indices}")
        images = self.load_property_images(property_name, photo_indices, resolution, base_path)
        
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

def test_pipeline():
    """
    Test the property analysis pipeline
    """
    # Initialize pipeline
    pipeline = PropertyAnalysisPipeline()
    
    # ===== INPUT PARAMETERS - MODIFY THESE =====
    property_name = "Dufaystraat 7-2"  # Change this to your property name
    photo_indices = [1, 3, 5, 8, 10]   # Change this to your photo numbers
    resolution = "360x240"              # Choose: "180x120" or "360x240"
    # ==========================================
    
    # Run analysis
    result = pipeline.analyze_property(
        property_name=property_name,
        photo_indices=photo_indices,
        resolution=resolution,
        base_path="./funda_images",
        metadata_file="Synthe_Loc_New/final_data_similar_filtered_synth_loc_added.json"
    )
    
    print("="*50)
    print("ANALYSIS RESULT:")
    print("="*50)
    print(result)
    
    return result

if __name__ == "__main__":
    test_pipeline()