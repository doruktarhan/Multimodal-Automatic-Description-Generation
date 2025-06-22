#preprocessor.py
import json
import yaml
import os
from typing import List, Dict, Any, Optional
import textwrap

class Preprocessor:
    """
    Transforms raw property data into a structured format for training.
    """
    def __init__(self):
        """
        Initialize the preprocessor and load configuration.
        """
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file is not found
            yaml.YAMLError: If config file is malformed
        """
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'preprocessor_config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def generate_chat_template(self, item: dict) -> dict[str, str]:
        """
        Generate chat template based on configuration.
        
        Args:
            item: Property item dictionary
            
        Returns:
            dict: Dictionary with system_prompt and user_prompt
        """
        prompts_config = self.config.get('prompts', {})
        
        system_prompt = prompts_config.get('system_prompt', '')
        user_prompt_template = prompts_config.get('user_prompt_template', '')
        
        # Prepare formatting parameters
        format_params = {
            'property_name': item.get('property_name', 'N/A'),
            'neighborhood': item.get('neighborhood', 'N/A'),
            'features_json': json.dumps(item.get('features', {}), indent=2)
        }
        
        # Try to add visual_cues if the template expects it
        try:
            # Test if the template contains visual_cues placeholder
            if '{visual_cues}' in user_prompt_template:
                format_params['visual_cues'] = item.get('visual_cues', 'N/A')
            
            # Format the user prompt template with the item data
            user_prompt = user_prompt_template.format(**format_params)
            
        except KeyError as e:
            # If template has placeholders we don't support, format with available params only
            user_prompt = user_prompt_template.format(
                property_name=format_params['property_name'],
                neighborhood=format_params['neighborhood'],
                features_json=format_params['features_json']
            )
        
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    
    def create_chat_example(self, item: dict) -> Optional[list[dict[str, str]]]:
        # Skip items without usable description
        desc = item.get("description", "")
        if not desc.strip():
            return None

        prompts   = self.generate_chat_template(item)
        # If you really want the header, add it here:
        # target = f"PROPERTY DESCRIPTION:\n{desc.rstrip()}"
        target = desc.rstrip()

        messages = [
            {"role": "system",    "content": prompts["system_prompt"]},
            {"role": "user",      "content": prompts["user_prompt"]},
            {"role": "assistant", "content": target}
        ]
        return messages


    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Process the entire dataset
        
        Args:
            data: List of property items
            
        Returns:
            List of processed examples with input and output text
        """
        processed_examples = []
        
        for item in data:
            example = self.create_chat_example(item)
            if example:  # Skip None results (items without descriptions)
                processed_examples.append(example)
        
        return processed_examples    


    def set_custom_prompt_template(self, template_function):
        """
        Set a custom prompt generation function
        
        Args:
            template_function: Function that takes extracted data and returns a prompt
        """
        self.generate_prompt = template_function