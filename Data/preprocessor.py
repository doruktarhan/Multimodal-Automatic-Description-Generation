#preprocessor.py
import json
from typing import List, Dict, Any
import textwrap

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
    

    def generate_chat_template(self, item: dict) -> dict[str, str]:
        system_prompt = (
            "You are a real‑estate agent responsible for writing property "
            "descriptions using the given metadata of the house.\n"
            "Always produce the following sections once and in this order:\n"
            "1. INTRODUCTION – brief overview\n"
            "2. LAYOUT – describe rooms and spaces\n"
            "3. LOCATION – info about the neighbourhood\n"
            "4. SPECIAL FEATURES – bullet list of key selling points\n\n"
            "Rules:\n"
            "• Mention only features present in the metadata.\n"
            "• Don’t invent amenities, locations or measurements if not necessary.\n"
            "• Start your answer after the literal text “PROPERTY DESCRIPTION:” "
            "and begin with “INTRODUCTION: ”."
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
    
    def create_chat_example(self, item: dict) -> list[dict[str, str]] | None:
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