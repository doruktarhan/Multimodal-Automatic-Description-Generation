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
        
        # Extract key factual elements that MUST be accurate
        room_info = features.get('Layout', {}).get('Number of rooms', 'N/A')
        living_area = features.get('Surface areas and volume', {}).get('Living area', 'N/A')
        year_built = features.get('Construction', {}).get('Year of construction', 'N/A')
        property_type = features.get('Construction', {}).get('Type apartment', 'N/A')
        raw_energy_label = features.get('Energy', {}).get('Energy label', 'N/A')
        if raw_energy_label and ' ' in raw_energy_label:
            energy_label = raw_energy_label.split(' ')[0]
        else:
            energy_label = raw_energy_label
        vve_info = features.get('VVE (Owners Association) checklist', {})
        vve_contribution = features.get('Transfer of ownership', {}).get('VVE (Owners Association) contribution', 'N/A')
        exterior_info = features.get('Exterior space', {})
        
        prompt = f"""You are a real estate agent writing comprehensive property descriptions. Create a detailed, engaging description for this property that incorporates necessary information from the provided metadata. Ensure absolute accuracy on key facts while highlighting all features and benefits of the property.

    Property: {property_name}
    Location: {neighborhood}, Amsterdam

    CRITICAL ACCURACY ELEMENTS - these MUST be presented exactly as stated:
    - Property type: {property_type}
    - Living area: {living_area}
    - Room configuration: {room_info}
    - Year built: {year_built}
    - Energy label: {energy_label}
    - Floor level: {features.get('Layout', {}).get('Located at', 'N/A')}
    - Ownership situation: {features.get('Cadastral data', {}).get('Ownership situation', 'N/A')}

    Property details:
    {json.dumps(features, indent=2)}

    Your description must incorporate ALL features from the metadata while maintaining this structure:
    1. INTRODUCTION (Brief overview highlighting the property's unique selling points)
    2. LAYOUT (Detailed description of all rooms, floors, and spaces exactly as presented in the data)
    3. LOCATION (Description of ONLY the specific neighborhood mentioned, without adding fictional proximity to other areas)
    4. SPECIAL FEATURES (Comprehensive list of notable features based ONLY on provided data)

    CRITICAL RULES:
    - Include ALL features from the metadata in your description
    - NEVER alter the number of bedrooms, bathrooms, or total rooms
    - NEVER invent amenities (fireplaces, walk-in closets, etc.) that aren't in the data
    - NEVER claim proximity to neighborhoods or landmarks not mentioned in the data
    - NEVER fabricate public transport lines or routes unless specifically stated
    - NEVER invent balcony/garden orientations if not specified
    - NEVER make claims about renovation unless explicitly stated
    - Keep floor plans and room arrangements exactly as described in the data
    - Maintain accuracy about VVE (homeowners' association) status and costs
    - Don't use disclaimers if it is not necessary.

    Generate a compelling, comprehensive description that showcases ALL features while maintaining absolute factual accuracy.
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