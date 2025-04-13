import json
import os
from google import genai
from dotenv import load_dotenv

import json

def calculate_gemini_price(response, model="gemini-2.0-flash"):
    """
    Calculate the price for a Gemini API call based on the API response metadata.
    
    Args:
        response: The Gemini API response object containing usage metadata
        model (str): The Gemini model used
        
    Returns:
        dict: Dictionary containing token counts and pricing information
    """
    # Define pricing based on the model
    pricing = {
        "gemini-2.0-flash": {
            "input_price_per_1k": 0.0001,  # $0.10 per 1K input tokens
            "output_price_per_1k": 0.0004  # $0.40 per 1K output tokens
        }
    }
    
    # Ensure model is supported
    if model not in pricing:
        raise ValueError(f"Model {model} pricing not defined")
    
    # Get token counts directly from the response metadata
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    
    # Calculate costs
    input_cost = (input_tokens / 1000) * pricing[model]["input_price_per_1k"]
    output_cost = (output_tokens / 1000) * pricing[model]["output_price_per_1k"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def print_pricing_summary(pricing_info, num_runs=1):
    """
    Print a summary of the pricing information
    
    Args:
        pricing_info (dict): The pricing information from calculate_gemini_price
        num_runs (int): Number of runs to calculate total for
    """
    print(f"\n--- Gemini API Pricing for {pricing_info['model']} ---")
    print(f"Input tokens: {pricing_info['input_tokens']} tokens")
    print(f"Output tokens: {pricing_info['output_tokens']} tokens")
    print(f"Input cost: ${pricing_info['input_cost']:.6f}")
    print(f"Output cost: ${pricing_info['output_cost']:.6f}")
    print(f"Total cost per run: ${pricing_info['total_cost']:.6f}")
    
    if num_runs > 1:
        total_cost_all_runs = pricing_info['total_cost'] * num_runs
        print(f"\nTotal cost for {num_runs} runs: ${total_cost_all_runs:.6f}")




load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


prompt = """
I will send you an address for a house in Amsterdam, and I want you to write me a good description of the surroundings of the address to include it in the house description. An example is shown below: 


Chasséstraat 20-3:

SURROUNDINGS & ACCESSIBILITY
The apartment is located on Chasséstraat, a wonderfully quiet street, while the direct surroundings offer a great variety of shops, supermarkets, restaurants, and nightlife venues, including De Hallen and the Ten Kate Market.
Just around the corner, you'll find Le French Café, De Neef van Fred, and Café Thuys, perfect spots for going out and socializing!
Within just a few minutes by bike, you can reach the city center, Museumplein, Leidseplein, or the greenery of Vondelpark. The accessibility is excellent.
The tram and bus stops at Admiraal de Ruijterweg and Postjesweg & Kinkerstraat are within walking distance, offering various lines, including a night bus. 
The A10 ring road can be reached within minutes, providing quick access to the A2, A4, and A9 motorways.
Parking is available with a permit in front of the building and in the parking garage at Piri Reisplein.


Now do the same for the following adress:  Joos de Moorstraat 29-2:
"""

print(prompt)

"""Make a request to the Gemini API with optimized parameters."""
response = client.models.generate_content(
    model=model,
    contents=[prompt],
    config=genai.types.GenerateContentConfig(
        temperature =  0.5,  # Deterministic output for classification
        max_output_tokens =  5000 # Just enough for single-word response with padding
    )
)

# Get token counts from response metadata
# response.usage_metadata.total_token_count
# response.usage_metadata.prompt_token_count
# response.usage_metadata.candidates_token_count

# Calculate pricing
pricing_info = calculate_gemini_price(response, model)
print_pricing_summary(pricing_info)
print_pricing_summary(pricing_info, num_runs=1000)