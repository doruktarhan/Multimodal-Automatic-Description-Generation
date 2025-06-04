import os
import pandas as pd
import json
from google import genai
from typing import Dict, List, Any, Optional, Tuple
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)

from typing import Dict, Optional, List
from vertexai.evaluation import (
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    EvalTask,
)

class CustomPointwiseMetric:
    """
    Convenience wrapper that builds a PointwiseMetricPromptTemplate
    and exposes a ready-to-use PointwiseMetric via .build().
    """

    def __init__(
        self,
        metric: str,
        description: str,
        criteria: Dict[str, str],
        rating_rubric: Dict[str, str],
        instruction: Optional[str] = None,

        input_variables: Optional[List[str]] = ["prompt", "reference", "response"]  ,
        metric_definition: Optional[str] = None,
        evaluation_steps: Optional[Dict[str, str]] = None,
    ):
        self.metric = metric
        self.description = description
        self.criteria = criteria
        self.rating_rubric = rating_rubric
        self.instruction = instruction
        self.input_variables = input_variables 
        self.metric_definition = metric_definition
        self.evaluation_steps = evaluation_steps

        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self) -> PointwiseMetricPromptTemplate:
        return PointwiseMetricPromptTemplate(
            criteria=self.criteria,
            rating_rubric=self.rating_rubric,
            instruction=self.instruction,
            input_variables=self.input_variables,      # <- key piece
            metric_definition=self.metric_definition,
            evaluation_steps=self.evaluation_steps,
        )

    def build(self) -> PointwiseMetric:
        """Return a PointwiseMetric ready for EvalTask."""
        metric_obj =  PointwiseMetric(
            metric=self.metric,
            metric_prompt_template=self.prompt_template,
        )
        # attach a callable that returns the template string
        #metric_obj.metric_prompt_template = lambda: str(self.prompt_template)
        return metric_obj



class EvalDatasetVertex:
    """
    Builds and validates a pandas-based evaluation dataset
    for BYOP (bring-your-own-prediction) evaluations.
    Required columns: prompt, response  (+ whatever your template needs).
    """

    def __init__(
        self,
        prompt_list: List[str],
        reference_list: List[str],
        response_list: List[str],
        **extra_cols,         # e.g. property_name=["id1", …]
    ):
        assert len(prompt_list) == len(reference_list) == len(response_list), \
            "All lists must have the same length"

        data = {
            "prompt": prompt_list,
            "reference": reference_list,
            "response": response_list,
        }
        # merge any user-supplied extra columns
        for col, values in extra_cols.items():
            assert len(values) == len(prompt_list), f"Column {col} length mismatch"
            data[col] = values

        self._df = pd.DataFrame(data)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def __len__(self):
        return len(self._df)







