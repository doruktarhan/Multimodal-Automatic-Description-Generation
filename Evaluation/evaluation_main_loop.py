import os
os.environ["CLOUD_ML_IPYTHON"] = "0"  
print("CLOUD_ML_IPYTHON early value:", os.getenv("CLOUD_ML_IPYTHON"))

import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock

# Mock the IPython modules
sys.modules['IPython'] = MagicMock()
sys.modules['IPython.core'] = MagicMock()
sys.modules['IPython.core.display'] = MagicMock()
sys.modules['IPython.display'] = MagicMock()

# Add display function to IPython.core.display
sys.modules['IPython.core.display'].display = MagicMock()

import json
import glob
import re
from pathlib import Path
from typing import Dict, List
import vertexai
from Data.preprocessor import Preprocessor
from Evaluation.vertex_evaluation import EvalDatasetVertex, CustomPointwiseMetric, EvalTask
from vertexai.generative_models import GenerativeModel
from vertexai.evaluation import PointwiseMetric,PointwiseMetricPromptTemplate, MetricPromptTemplateExamples

from dotenv import load_dotenv
#gemini api key
load_dotenv()

vertexai.init(
    project = os.getenv("GEMINI_PROJECT_ID"),
    location=os.getenv("GEMINI_LOCATION")
)

preprocessor = Preprocessor()

evaluator_model = GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"temperature": 0.1, "max_output_tokens": 1000, "top_k": 1},
)

# Load the datasets
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to create a valid Vertex AI experiment name
def create_valid_experiment_name(name):
    # Convert to lowercase
    name = name.lower()
    
    # Replace underscores with hyphens
    name = name.replace("_", "-")
    
    # Remove any characters that aren't lowercase letters, numbers, or hyphens
    name = re.sub(r'[^a-z0-9-]', '', name)
    
    # Ensure it starts with a letter or number (not a hyphen)
    if name and name[0] == '-':
        name = 'exp' + name
    elif not name:
        name = 'experiment'
        
    # Truncate if longer than 128 characters
    if len(name) > 128:
        name = name[:128]
        
    return name

test_data_path = "Data/test_data.json"
test_data = load_json(test_data_path)

# Base directory containing model folders (up to the LLM_7B part)
base_dir = "/Users/doruktarhan/Desktop/Master Thesis Model Runs/LLM_7B_Instruct_Chat"

# Extract model type from base directory (LLM_7B_Instruct_Chat)
model_type = os.path.basename(base_dir)
print(f"Extracted model type: {model_type}")

# Navigate to the Inference_output directory
inference_output_dir = os.path.join(base_dir, "Inference_output")
if not os.path.exists(inference_output_dir):
    print(f"Error: Inference output directory not found at {inference_output_dir}")
    sys.exit(1)

# Find the experiment directory (there should be only one)
experiment_dirs = glob.glob(os.path.join(inference_output_dir, "saved_model_*"))
if not experiment_dirs:
    print(f"Error: No experiment directories found in {inference_output_dir}")
    sys.exit(1)

experiment_dir = experiment_dirs[0]  # Take the first one if there are multiple
experiment_base_name = os.path.basename(experiment_dir)
print(f"Using experiment directory: {experiment_dir}")

# Find all model directories that might contain generated_samples.json
model_dirs = []
for item in glob.glob(os.path.join(experiment_dir, "*")):
    if os.path.isdir(item):
        # Check if generated_samples.json exists in this directory
        if os.path.exists(os.path.join(item, "generated_samples.json")):
            model_dirs.append(item)

if not model_dirs:
    print(f"No model directories with generated_samples.json found in {experiment_dir}")
else:
    print(f"Found {len(model_dirs)} model directories to evaluate:")
    for dir_path in model_dirs:
        print(f"  - {os.path.basename(dir_path)}")
    
    # Loop through each model directory
    for model_dir in model_dirs:
        model_folder_name = os.path.basename(model_dir)
        print(f"\n{'='*80}\nProcessing model: {model_folder_name}\n{'='*80}")
        
        # Set the current model's generated_samples path
        generated_samples_path = os.path.join(model_dir, "generated_samples.json")
        print(f"Loading generated samples from: {generated_samples_path}")
        
        try:
            # Create experiment name in a valid format for Vertex AI
            # Use model type and model folder name to create a unique identifier
            raw_experiment_name = f"{model_type}-{model_folder_name}"
            experiment_name = create_valid_experiment_name(raw_experiment_name)
            print(f"Using experiment name: {experiment_name}")
            
            # Load the generated samples for this model
            generated_samples = load_json(generated_samples_path)
            
            #create a lookup table for the test_data
            test_data_lookup = {item['property_name']: item for item in test_data}
            required_keys = ["neighborhood", "features", "description"]
            
            evaluation_pairs = []
            for property_name, generated_description in generated_samples['results'].items():
                # Find the original test data for this property
                if property_name in test_data_lookup:
                    original_data = test_data_lookup[property_name]
                    
                    # Extract the relevant information
                    evaluation_pair = {
                        "property_name": property_name,
                        "neighborhood": original_data.get("neighborhood", ""),
                        "features": original_data["features"],
                        "original_description": original_data["description"],
                        "generated_description": generated_description,
                        "url": original_data.get("url", "")
                    }
                    
                    evaluation_pairs.append(evaluation_pair)
                else:
                    print(f"Warning: No test data found for property {property_name}")
            
            print(f"Found {len(evaluation_pairs)} complete properties for evaluation")
            
            if not evaluation_pairs:
                print(f"No evaluation pairs found for {model_folder_name}, skipping...")
                continue
                
            prep = Preprocessor()          
            
            prompt_list    : list[str] = []
            reference_list : list[str] = []      # the ground-truth description
            response_list  : list[str] = []      # the model-generated text
            property_names : list[str] = []      # optional: keep an ID column
            
            for pair in evaluation_pairs:            # built in your earlier script
                # ---- 1️⃣ build the user prompt that *caused* the model output ----
                prompt = prep.generate_prompt({
                    "property_name": pair["property_name"],
                    "neighborhood" : pair["neighborhood"],
                    "features"     : pair["features"],
                })
            
                # ---- 2️⃣ grab reference & model response ----
                reference = pair["original_description"]
                response  = pair["generated_description"]
            
                # ---- 3️⃣ collect ----
                prompt_list.append(prompt)
                reference_list.append(reference)
                response_list.append(response)
                property_names.append(pair["property_name"])
            
            print(f"Collected {len(prompt_list)} evaluation pairs")
            print(f"Property Name: {property_names[0]}")
            
            
            # create eval dataset dataframe suitable for vertex
            eval_ds = EvalDatasetVertex(
                prompt_list=prompt_list,
                reference_list=reference_list,
                response_list=response_list,
                property_name=property_names,    
            ).dataframe
            
            
            #test dummy variable 
            criteria = {
                "factual_accuracy": "The description accurately represents the property's features.",
                "completeness": "The description includes all important property features.",
                "fluency": "The description is well-written and flows naturally."
            }
            
            rating_rubric = {
                "5": "(Very good) Excellent on all criteria.",
                "4": "(Good) Strong performance on most criteria.",
                "3": "(Average) Acceptable performance with some issues.",
                "2": "(Poor) Significant issues in multiple areas.",
                "1": "(Very poor) Major problems in most areas."
            }
            
            instruction = ( 
                "You are an evaluator for property descriptions. "
                "Your task is to rate the generated description based on the given criteria."
            )
            
            dummy_metric = CustomPointwiseMetric(
                metric="dummy_metric",
                description="Dummy metric for testing purposes.",
                criteria=criteria,
                rating_rubric=rating_rubric,
                instruction=instruction,
                input_variables=["prompt", "reference", "response"],
                metric_definition="This is a dummy metric for testing.",
                #evaluation_steps=["Evaluate the response based on the criteria."]
            ).build()
            
            
            #get a built in metric
            fluency_metric = MetricPromptTemplateExamples.Pointwise.FLUENCY
            verbose_metric = MetricPromptTemplateExamples.Pointwise.VERBOSITY
            
            
            #create the eval task
            eval_task = EvalTask(
                dataset = eval_ds,
                metrics = [dummy_metric, fluency_metric, verbose_metric],
                experiment = experiment_name
                )
            
            #evaluate metrics
            print(f"Running evaluation for {experiment_name}...")
            evaluation_results = eval_task.evaluate()
            
            
            #############################################################
            ########## Print and save the evaluation results
            #############################################################
            
            # Create model-specific output directory
            model_out_dir = Path(model_dir) / "evaluation_outputs"
            model_out_dir.mkdir(exist_ok=True)
            
            # -------- 1️⃣  quick on-screen summary --------
            print("\n===   Experiment:", evaluation_results.metadata["experiment"], "===")
            for name, value in evaluation_results.summary_metrics.items():
                print(f"{name:<30s} : {value:5.3f}")
            
            # -------- 2️⃣  save the per-row table --------
            csv_path = model_out_dir / "metrics_table.csv"
            evaluation_results.metrics_table.to_csv(csv_path, index=False)
            
            print(f"\nPer-row metrics written to {csv_path.resolve()}")
            
            # -------- 3️⃣  save summary + metadata --------
            summary_path = model_out_dir / "summary_metrics.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary_metrics": evaluation_results.summary_metrics,
                        "metadata": evaluation_results.metadata,
                    },
                    f,
                    indent=2,
                )
            print(f"Run summary saved to {summary_path.resolve()}")
            
        except Exception as e:
            print(f"Error processing {model_folder_name}: {str(e)}")
            
    print("\nAll model evaluations completed!")
    
    # Create a combined summary at the end (optional)
    combined_summary = {}
    for model_dir in model_dirs:
        model_folder_name = os.path.basename(model_dir)
        summary_path = os.path.join(model_dir, "evaluation_outputs", "summary_metrics.json")
        
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    model_summary = json.load(f)
                combined_summary[model_folder_name] = model_summary.get("summary_metrics", {})
            except Exception as e:
                print(f"Error loading summary for {model_folder_name}: {str(e)}")
    
    if combined_summary:
        combined_path = os.path.join(experiment_dir, "combined_evaluation_summary.json")
        with open(combined_path, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"Combined summary saved to {combined_path}")