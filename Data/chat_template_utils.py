# chat_template_utils.py
from typing import List, Dict, Any
import torch
import re

def is_qwen_text_model(model_name_or_path):
    """
    Check if the model is a Qwen text (pure decoder) model.
    
    Args:
        model_name_or_path: The model name or path
        
    Returns:
        bool: True if it's a Qwen text model, False otherwise
    """
    # Check if the model name contains 'Qwen' but not 'VL'
    # This covers Qwen, Qwen2, etc. but excludes Qwen-VL variants
    if isinstance(model_name_or_path, str):
        model_name_lower = model_name_or_path.lower()
        return 'qwen' in model_name_lower and 'vl' not in model_name_lower
    return False

def get_qwen_chat_template():
    """
    Returns the modified Qwen chat template with correctly placed generation tags.
    
    This template handles both training (with assistant content) and inference
    (with generation prompt) scenarios.
    
    Returns:
        str: The modified chat template as a string
    """
    # The template places {% generation %} and {% endgeneration %} tags outside the Jinja2 output blocks
    modified_template = """
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
    {% generation %}    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}    {% endgeneration %}
    {%- elif message.role == "assistant" %}
        {% generation %}{{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\\\"name\\\": \\\"' }}
            {{- tool_call.name }}
            {{- '\\\", \\\"arguments\\\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {% generation %}{{- '<|im_start|>assistant\\n' }}{% endgeneration %}
{%- endif %}
"""
    return modified_template

def initialize_chat_template(tokenizer):
    """
    Initialize the chat template based on tokenizer model type.
    Sets the appropriate template for Qwen text models.
    
    Args:
        tokenizer: The tokenizer to check and update
        
    Returns:
        bool: True if a custom template was applied, False otherwise
    """
    if hasattr(tokenizer, 'name_or_path') and is_qwen_text_model(tokenizer.name_or_path):
        # Store the original template if needed later
        original_template = tokenizer.chat_template
        # Set the Qwen-specific template
        tokenizer.chat_template = get_qwen_chat_template()
        print(f"Set Qwen-specific chat template for {tokenizer.name_or_path}")
        return True
    return False