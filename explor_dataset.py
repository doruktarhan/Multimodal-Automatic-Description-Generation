# explore_data.py
import json
import os

def load_json_data(file_path):
    """Load JSON data from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_prompt_from_data(property_data):
    """
    Build a prompt using property details.
    """
    features = property_data.get('features', {})
    neighborhood = property_data.get('neighborhood', '')
    property_name = property_data.get('property_name', '')
    
    prompt = f"""You are a real estate agent writing property descriptions. Create a natural, engaging description for this house based on the following data. Focus on the key selling points and maintain a professional tone.

                Property: {property_name}
                Location: {neighborhood}

                Property details:
                {json.dumps(features, indent=2)}

                Write a compelling property description that highlights the main features and benefits."""
                    
    return prompt


def explore_data(data):
    """Print basic information about the dataset"""
    print(f"Dataset contains {len(data)} samples")
    
    # Look at the first sample
    if data:
        first_sample = data[0]
        print("\nKeys in first sample:")
        for key in first_sample.keys():
            print(f"- {key}")
        
        # Check if 'features' exists and what it contains
        if 'features' in first_sample:
            print("\nFeature categories:")
            for category in first_sample['features'].keys():
                print(f"- {category}")
        
        # Print description length
        if 'description' in first_sample:
            desc_length = len(first_sample['description'])
            print(f"\nDescription length: {desc_length} characters")
            print(f"Description preview: {first_sample['description'][:200]}...")

    # Add prompt to each item in the dataset
    for item in data:
        item['prompt'] = generate_prompt_from_data(item)

    # Average description length
    total_length = sum(len(item['description']) for item in data)
    average_length = total_length / len(data)
    print(f"\nAverage description length: {average_length:.2f} characters")

    # Average prompt length
    total_prompt_length = sum(len(item['prompt']) for item in data)
    average_prompt_length = total_prompt_length / len(data)
    print(f"\nAverage prompt length: {average_prompt_length:.2f} characters")


data_path = "SFT_Inference_CodeBase/Data/funda_scrapped_amsterdam_sample.json"

# Check if file exists
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit()
# Load and explore data
data = load_json_data(data_path)
explore_data(data)

# Optionally save the updated data with prompts
# output_path = "SFT_Inference_CodeBase/Data/funda_scrapped_amsterdam_sample_with_prompts.json"
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(data, f, indent=2)
# print(f"\nUpdated data saved to {output_path}")
