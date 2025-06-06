#preprocessor.py
import json
from typing import List, Dict, Any, Optional
import textwrap

class Preprocessor:
    """
    Transforms raw property data into a structured format for training.
    """
    def __init__(self,):

        """
        Initialize the preprocessor with the maximum input and output lengths.
        """


    # def generate_chat_template(self, item: dict) -> dict[str, str]:
    #     system_prompt = (
    #         "You are a real‑estate agent responsible for writing property "
    #         "descriptions using the given metadata of the house. Write in an engaging, "
    #         "professional tone that highlights the property's best features.\n"
    #         "Always produce the following sections once and in this order:\n"
    #         "1. INTRODUCTION – brief overview\n"
    #         "2. LAYOUT – describe rooms and spaces\n"
    #         "3. LOCATION – info about the neighbourhood (stay general unless specific details are provided)\n"
    #         "4. SPECIAL FEATURES – bullet list of key selling points with brief value descriptions\n\n"
    #         "Rules:\n"
    #         "• Mention only features present in the metadata.\n"
    #         "• Don't invent amenities, locations or measurements if not mentioned in te tabular data.\n"
    #         "• You may use descriptive adjectives for verified features (e.g., 'spacious' for large areas).\n"
    #         "• Start your answer after the literal text “PROPERTY DESCRIPTION:” "
    #         "and begin with “INTRODUCTION: ”."
    #     )

    #     raw_user = f"""
    #     Property name: {item.get('property_name', 'N/A')}
    #     Neighbourhood: {item.get('neighborhood', 'N/A')}

    #     FEATURES JSON:
    #     {json.dumps(item.get('features', {}), indent=2)}

    #     Write a compelling property description that highlights the main features and benefits below.
    #     """

    #     user_prompt = textwrap.dedent(raw_user).strip()

    #     return {"system_prompt": system_prompt, "user_prompt": user_prompt}


################################## LESS RESTRICTED PROMPT ##########################################################################
##############################################################################################################################

    def generate_chat_template(self, item: dict) -> dict[str, str]:

        system_prompt = (
            "You are a skilled real-estate copywriter. Your primary goal is to craft a professional, engaging, and "
            "vivid property description using the provided metadata. Aim to highlight the property's best features "
            "and its overall appeal to potential buyers or renters, making the most of the information available.\n\n"
            "Always produce the following sections once and in this order:\n"
            "1. INTRODUCTION – Provide a concise and inviting overview of the property, capturing its essence and key appeal based on the metadata.\n"
            "2. LAYOUT – Clearly describe the rooms, spaces, and their arrangement as detailed in the metadata. Use descriptive language to "
            "   convey their atmosphere and functionality, elaborating on the listed features.\n"
            "3. LOCATION – Detail the property's location and neighborhood character *based on the 'Neighbourhood' information "
            "   and any related location details explicitly provided in the metadata*. If specific neighborhood amenities or "
            "   characteristics are listed in the metadata, elaborate on these to showcase their benefits. If the provided neighborhood "
            "   data is general or minimal, focus on describing the property's setting (e.g., 'situated in a residential area,' "
            "   'urban apartment living') as supported by the input, or its general accessibility if such information is available.\n"
            "4. SPECIAL FEATURES – Present a bulleted list of notable selling points taken directly from the metadata. For each point, "
            "   briefly explain its value or benefit to enhance its attractiveness, using the provided details as your foundation.\n\n"
            "Guidelines for Content Generation:\n"
            "• Adherence to Metadata: Ensure that all factual claims (number of rooms, sizes, specific amenities like 'garage', 'balcony', "
            "  energy labels, year built) are directly and accurately supported by the provided metadata.\n"
            "• Engaging and Descriptive Language: You are strongly encouraged to use rich adjectives, professional vocabulary, and "
            "  descriptive phrasing to make the *features and details explicitly mentioned in the metadata* sound appealing and to "
            "  vividly convey a sense of the property's atmosphere and benefits. For example, if the metadata lists a 'garden', "
            "  you can describe it as 'a delightful garden space, perfect for outdoor relaxation.'\n"
            "• Professional Tone: Maintain a professional, clear, positive, and trustworthy tone throughout the description.\n"
            "• Avoid Inventing Specifics: Do not invent specific amenities, unique architectural details, room dimensions, "
            "  or detailed neighborhood characteristics (like names of specific local cafes, shops, exact unlisted distances, "
            "  or unique neighborhood events) *if they are not present in the metadata*. Your creativity should focus on "
            "  embellishing and presenting the *provided information* in the best possible light, not on adding new, unverified 'facts'.\n"
            "• Starting Phrase: Always start your entire answer after the literal text “PROPERTY DESCRIPTION:” "
            "  and begin the first section with “INTRODUCTION: ” (followed by your text)."
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


################################## FREE ROAM PROMPT ##########################################################################
##############################################################################################################################

    # def generate_chat_template(self, item: dict) -> dict[str, str]:

    #     system_prompt = (
    #         "You are an expert real estate copywriter located in Amsterdam. Your task is to create a single, "
    #         "compelling, and informative property description based on the provided metadata. "
    #         "Your main goal is to engage potential buyers or renters and highlight what makes the property attractive.\n\n"
    #         "Use the metadata as your factual basis. You have creative freedom in how you structure the description, "
    #         "the language you use, and the aspects you choose to emphasize to best showcase the property and its "
    #         "location (if neighborhood information is provided in the metadata).\n\n"
    #         "Aim for a professional, positive, and persuasive tone. Ensure the core factual details from the metadata "
    #         "(like number of rooms, essential features) are naturally woven into your narrative. Avoid inventing "
    #         "specific measurements or highly unique amenities if they are not mentioned in the metadata."
    #     )
        
    #     raw_user = f"""
    #     Property name: {item.get('property_name', 'N/A')}
    #     Neighbourhood: {item.get('neighborhood', 'N/A')}

    #     FEATURES JSON:
    #     {json.dumps(item.get('features', {}), indent=2)}

    #     Write a compelling property description that highlights the main features and benefits below.
    #     """

    #     user_prompt = textwrap.dedent(raw_user).strip()

    #     return {"system_prompt": system_prompt, "user_prompt": user_prompt}

#######################################################################################################################
    
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