import json
import time
import torch
import psutil
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

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


def main():
    # Get Hugging Face token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")

    # Load the Qwen2.5-3B-Instruct model and tokenizer from Hugging Face with token
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load property data (assumed JSON structure as in your original code)
    with open('SFT_Inference_CodeBase/Data/funda_scrapped_amsterdam_sample.json', 'r') as file:
        properties = json.load(file)
    
    results = []
    
    for i, property_data in enumerate(properties):
        try:
            # Generate the user prompt based on property data
            user_prompt = generate_prompt_from_data(property_data)
            system_message = "You are a professional real estate agent you writes property descriptions."
            
            # Construct messages in the expected format for Qwen2.5
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply the chat template (this formats the conversation into a single prompt)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize the formatted text and move it to the model's device
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # Generate a response from the model with a token limit (max_new_tokens)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=500
            )
            
            # Remove the input part of the sequence to isolate the generated output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode the tokens to get the final text output
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Build the result entry (combining original property info with the generated description)
            result = {
                "url": property_data.get("url", ""),
                "neighborhood": property_data.get("neighborhood", ""),
                "property_name": property_data.get("property_name", ""),
                "original_description": property_data.get("description", ""),
                "generated_description": response
            }
            results.append(result)
            
            print(f"Processed {i+1}/{len(properties)} properties.")
            
            
        except Exception as e:
            print(f"Error processing property {property_data.get('property_name', 'unknown')}: {str(e)}")
        
    # Save the results into a new JSON file
    with open('automatic_description_Qwen2.5_30_samples.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)
        
    print("Processing complete. Results saved to automatic_description_Qwen2.5_30_samples.json")

if __name__ == "__main__":
    main()
