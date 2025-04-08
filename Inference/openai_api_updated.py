import json
import openai
from openai import OpenAI
import time

def generate_prompt_from_data(property_data):
    # Existing prompt generation logic remains the same
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

def main():
    # Initialize OpenAI client
    client = OpenAI()
    
    # Read the input JSON file
    with open('SFT_Inference_CodeBase/Data/funda_scrapped_amsterdam_sample.json', 'r') as file:
        properties = json.load(file)
    
    # Initialize results dictionary
    results = []
    
    i = 0
    # Process each property
    for property_data in properties:
        try:
            # Generate prompt for current property
            prompt = generate_prompt_from_data(property_data)
            
            # Make API call
            response = client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": "You are a professional real estate agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract the generated description
            generated_description = response.choices[0].message.content
            
            # Create result entry
            result = {
                "url": property_data["url"],
                "neighborhood": property_data["neighborhood"],
                "property_name": property_data["property_name"],
                "original_description": property_data["description"],
                "generated_description": generated_description
            }
            
            results.append(result)
            print(f"Processed {i+1} properties")
            i += 1
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing property {property_data.get('property_name', 'unknown')}: {str(e)}")

    
    # Save results to new JSON file
    with open('automatic_description_4o_30_samples.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)
        
    print(f"Processing complete. Results saved to automatic_description_4o_30_samples.json")
if __name__ == "__main__":
    main()
