"""Type definitions for Hugging Face to Lilypad converter"""

from .model_types import ModelType, detect_model_type
from .task_types import TaskType, detect_task_type

__all__ = ['ModelType', 'TaskType', 'detect_model_type', 'detect_task_type']