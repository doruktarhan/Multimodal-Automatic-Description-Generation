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
experiment_name = "dummy-test"


# evaluator_model = GenerativeModel(
#         "gemini-2.0-flash",
#         generation_config={"temperature": 0.1, "max_output_tokens": 1000, "top_k": 1},
# )

# Load the datasets
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

test_data_path = "Data/test_data.json"
generated_samples_path = "/Users/doruktarhan/Desktop/Master Thesis Model Runs/LLM_7B_Instruct_Chat/Inference_output/saved_model_qwen7b_1e-06_32_0_20250420_230903/base_model/generated_samples.json"


test_data = load_json(test_data_path)
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
print(f"Prompt: {prompt_list[0]}")
print(f"Reference: {reference_list[0]}")
print(f"Response: {response_list[0]}")
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
    metrics = [dummy_metric,fluency_metric, verbose_metric],
    experiment = experiment_name
    )

#evaluate metrics
evaluation_results = eval_task.evaluate(
    #model=evaluator_model
)
 





#############################################################
########## Print the evaluation results
#############################################################

# -------- 1️⃣  quick on-screen summary --------
print("\n===   Experiment:", evaluation_results.metadata["experiment"], "===")
for name, value in evaluation_results.summary_metrics.items():
    print(f"{name:<30s} : {value:5.3f}")

# -------- 2️⃣  save the per-row table --------
out_dir = Path("evaluation_outputs")        # folder will be created if absent
out_dir.mkdir(exist_ok=True)

csv_path = out_dir / "metrics_table.csv"
evaluation_results.metrics_table.to_csv(csv_path, index=False)

print(f"\nPer-row metrics written to {csv_path.resolve()}")

# -------- 3️⃣  save summary + metadata --------
summary_path = out_dir / "summary_metrics.json"
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

# -------- 4️⃣  quick look at weakest rows (optional) --------
METRIC_TO_INSPECT = "dummy_metric"          # change to any column in the CSV
df = evaluation_results.metrics_table

if METRIC_TO_INSPECT in df.columns:
    worst = df.nsmallest(5, METRIC_TO_INSPECT)[
        ["property_name", "prompt", "reference", "response", METRIC_TO_INSPECT]
    ]
    print("\n--- Five lowest-scoring rows on", METRIC_TO_INSPECT, "---")
    # Show just the score and a snippet of the response
    for _, row in worst.iterrows():
        snippet = row["response"][:120].replace("\n", " ") + "..."
        print(f"{row['property_name']:<25s} | {row[METRIC_TO_INSPECT]:4.2f} | {snippet}")
else:
    print(f"\nMetric '{METRIC_TO_INSPECT}' not found in metrics_table columns:", df.columns.tolist())