import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
from typing import List
from tqdm import tqdm

class EmbeddingExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        model_name = "Salesforce/blip-itm-large-coco"
        print(f"Loading BLIP model: {model_name}")
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded successfully!")
    
    def extract_image_embeddings(self, image_paths: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if images:
                dummy_text = [""] * len(images)
                inputs = self.processor(images=images, text=dummy_text, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        use_itm_head=False,
                        return_dict=True
                    )
                    
                    vision_embeds = outputs.last_hidden_state[:, 0, :]
                    proj_image_embedding = self.model.vision_proj(vision_embeds)
                    proj_image_embedding = proj_image_embedding / proj_image_embedding.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(proj_image_embedding.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for text captions"""
        text_embeddings = []
        
        for text in tqdm(texts, desc="Extracting text embeddings"):
            dummy_image = Image.new('RGB', (224, 224), color='white')
            inputs = self.processor(images=dummy_image, text=text, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                    use_itm_head=False,
                    return_dict=True
                )
                
                text_embeds = outputs.question_embeds[:, 0, :]
                proj_text_embedding = self.model.text_proj(text_embeds)
                proj_text_embedding = proj_text_embedding / proj_text_embedding.norm(p=2, dim=-1, keepdim=True)
                text_embeddings.append(proj_text_embedding.cpu().numpy())
        
        return np.vstack(text_embeddings)