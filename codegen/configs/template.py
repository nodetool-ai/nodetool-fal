"""
Configuration template for FAL node modules.

Copy this file and customize for each module.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    # Example:
    # "fal-ai/example-endpoint": {
    #     "class_name": "ExampleNode",
    #     "docstring": "Brief description of what this node does.",
    #     "tags": ["tag1", "tag2", "category"],
    #     "use_cases": [
    #         "Use case 1",
    #         "Use case 2",
    #         "Use case 3",
    #         "Use case 4",
    #         "Use case 5"
    #     ],
    #     "field_overrides": {
    #         "field_name": {
    #             "python_type": "str",  # Override type
    #             "default_value": '""',  # Override default
    #             "description": "Custom description"  # Override description
    #         }
    #     },
    #     "custom_imports": [
    #         "from custom.module import CustomClass"
    #     ],
    #     "custom_methods": {
    #         "process": '''
    #             # Custom process method implementation
    #             async def process(self, context: ProcessingContext) -> OutputType:
    #                 # Custom logic here
    #                 pass
    #         '''
    #     }
    # },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """
    Get configuration for an endpoint.
    
    Args:
        endpoint_id: FAL endpoint ID
        
    Returns:
        Configuration dictionary
    """
    return CONFIGS.get(endpoint_id, {})
