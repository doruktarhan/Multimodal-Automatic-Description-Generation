#preprocessor.py
import json
from typing import List, Dict, Any

class Preprocessor:
    """
    Transforms raw property data into a structured format for training.
    """
    def __init__(self,):

        """
        Initialize the preprocessor with the maximum input and output lengths.
        """


    def generate_prompt(self, item: Dict[str, Any]) -> str:
        features = item.get('features', {})
        neighborhood = item.get('neighborhood', 'N/A')
        property_name = item.get('property_name', 'N/A')
        
        prompt = f"""You are a real estate agent writing a property description for {property_name} in {neighborhood}, Amsterdam. Your job is to create a property description using the provided metadata for the property. 

    Create a professional property description with these sections:
    1. INTRODUCTION - Brief overview of the property
    2. LAYOUT - Description of the rooms and spaces
    3. LOCATION - Information about the {neighborhood} area
    4. SPECIAL FEATURES - Bullet list of key selling points, not all features.

    PROPERTY DETAILS:
    {json.dumps(features, indent=2)}

    Rules: Only mention features in the data. Don't invent amenities, locations, or claim proximity to areas not mentioned. Be accurate about room counts and measurements.
    You need to have exactly 1 of each sections. Start description after "PROPERTY DESCRIPTION: with "INTRODUCTION: "

    PROPERTY DESCRIPTION: 
    """
        return prompt
    
    def create_example(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a single training example from a property item
        
        Args:
            item: Property data dictionary
            
        Returns:
            Dictionary with input and output text
        """
        # Skip items without descriptions
        if 'description' not in item or not item['description'].strip():
            return None
        
        #Generate prompt
        input_text = self.generate_prompt(item)
        output_text = item.get('description', '')
        
        return {
            "input": input_text,
            "output": output_text
        }


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
            example = self.create_example(item)
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