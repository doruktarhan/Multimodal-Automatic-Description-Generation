from Evaluation_Config.vertex_evaluation import CustomPointwiseMetric

class MetricsRegistry:
    """Registry for storing and retrieving evaluation metrics."""
    
    def __init__(self):
        self._metrics = {}
        
    def register(self, metric_name, metric_obj):
        """Register a new metric."""
        self._metrics[metric_name] = metric_obj
        return self  # For method chaining
        
    def get_metric(self, metric_name):
        """Retrieve a registered metric by name."""
        if metric_name not in self._metrics:
            raise KeyError(f"Metric '{metric_name}' not registered")
        return self._metrics[metric_name]
        
    def get_all_metrics(self):
        """Get all registered metrics."""
        return list(self._metrics.values())
        
    def list_metrics(self):
        """List names of all available metrics."""
        return list(self._metrics.keys())


class MetricFactory:
    """Factory for creating metric objects from configurations."""
    
    @staticmethod
    def create_from_config(config):
        """Create a metric object from a configuration dict."""
        metric_params = {
            "metric": config["name"],
            "description": config["description"],
            "criteria": config["criteria"],
            "rating_rubric": config["rating_rubric"],
            "instruction": config.get("instruction"),
            "input_variables": config.get("input_variables", ["prompt", "reference", "response"]),
            "metric_definition": config.get("metric_definition"),
            "evaluation_steps": config.get("evaluation_steps"),
        }
        
        return CustomPointwiseMetric(**metric_params).build()