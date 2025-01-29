"""Template utilities for Lilypad modules"""

import os
import json
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

class TemplateManager:
    """Manages module templates"""
    
    def __init__(self, template_dir: str = "templates"):
        """Initialize template manager
        
        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
    def render_dockerfile(
        self,
        model_type: Dict[str, Any],
        requirements: List[str],
        system_packages: Optional[List[str]] = None
    ) -> str:
        """Render Dockerfile template
        
        Args:
            model_type: Model type information
            requirements: Python package requirements
            system_packages: System package requirements
            
        Returns:
            Rendered Dockerfile content
        """
        template = self.env.get_template("Dockerfile.jinja2")
        return template.render(
            model_type=model_type,
            requirements=requirements,
            system_packages=system_packages or []
        )
        
    def render_inference_script(
        self,
        model_type: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> str:
        """Render inference script template
        
        Args:
            model_type: Model type information
            model_config: Model configuration
            
        Returns:
            Rendered inference script
        """
        template_name = f"{model_type['task']}_inference.py.jinja2"
        template = self.env.get_template(template_name)
        return template.render(
            model_type=model_type,
            config=model_config
        )
        
    def render_module_config(
        self,
        model_type: Dict[str, Any],
        docker_image: str,
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """Render Lilypad module config template
        
        Args:
            model_type: Model type information
            docker_image: Docker image name
            env_vars: Additional environment variables
            
        Returns:
            Rendered module config
        """
        template = self.env.get_template("lilypad_module.json.jinja2")
        return template.render(
            model_type=model_type,
            docker_image=docker_image,
            env_vars=env_vars or {}
        )
        
    def render_readme(
        self,
        model_type: Dict[str, Any],
        model_info: Dict[str, Any],
        usage_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Render README template
        
        Args:
            model_type: Model type information
            model_info: Model metadata
            usage_examples: Example usage scenarios
            
        Returns:
            Rendered README
        """
        template = self.env.get_template("README.md.jinja2")
        return template.render(
            model_type=model_type,
            model_info=model_info,
            usage_examples=usage_examples or []
        )

    def render_test_script(
        self,
        model_type: Dict[str, Any],
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Render test script template
        
        Args:
            model_type: Model type information
            test_cases: Test case definitions
            
        Returns:
            Rendered test script
        """
        template = self.env.get_template("test_script.sh.jinja2")
        return template.render(
            model_type=model_type,
            test_cases=test_cases or []
        )

def create_default_templates(template_dir: str) -> None:
    """Create default template files if they don't exist
    
    Args:
        template_dir: Directory to create templates in
    """
    # Create template directory if it doesn't exist
    os.makedirs(template_dir, exist_ok=True)
    
    templates = {
        "Dockerfile.jinja2": '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    {% for package in system_packages %}
    {{ package }} {% if not loop.last %}\\ {% endif %}
    {% endfor %}
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    {% for req in requirements %}
    {{ req }} {% if not loop.last %}\\ {% endif %}
    {% endfor %}

# Create directories
RUN mkdir -p /cache/huggingface
RUN mkdir -p /outputs

# Set environment variables
ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1

# Copy model files
COPY ./model /app/model

# Copy inference script
COPY run_inference.py /app/

# Set output directory as volume
VOLUME /outputs

# Run inference script
CMD ["python", "/app/run_inference.py"]
''',
        
        "lilypad_module.json.jinja2": '''
{
    "machine": {
        "gpu": {% if model_type.requires_gpu %}1{% else %}0{% endif %},
        "cpu": 1000,
        "ram": {{ model_type.memory_requirements }}
    },
    "job": {
        "APIVersion": "V1beta1",
        "Spec": {
            "Deal": {
                "Concurrency": 1
            },
            "Docker": {
                "Entrypoint": ["python", "/app/run_inference.py"],
                "WorkingDirectory": "/app",
                "EnvironmentVariables": [
                    "HF_HUB_OFFLINE=1",
                    "TRANSFORMERS_OFFLINE=1"{% for key, value in env_vars.items() %},
                    "{{ key }}={{ value }}"{% endfor %}
                ],
                "Image": "{{ docker_image }}"
            },
            "Engine": "Docker",
            "Network": {
                "Type": "None"
            },
            "Outputs": [
                {
                    "Name": "outputs",
                    "Path": "/outputs"
                }
            ],
            "PublisherSpec": {
                "Type": "ipfs"
            },
            "Resources": {
                "GPU": "{% if model_type.requires_gpu %}1{% else %}0{% endif %}"
            },
            "Timeout": 1800
        }
    }
}
''',
        
        "README.md.jinja2": '''
# {{ model_info.name }} Lilypad Module

This module provides [{{ model_info.name }}]({{ model_info.url }}) for use on the Lilypad network.

## Model Details

- **Task**: {{ model_type.task }}
- **Input Type**: {{ model_type.input_type }}
- **Output Type**: {{ model_type.output_type }}
- **Supported Tasks**: {{ model_type.supported_tasks | join(", ") }}

{{ model_info.description }}

## Hardware Requirements

- GPU: {% if model_type.requires_gpu %}Required (minimum {{ model_type.memory_requirements }}MB VRAM){% else %}Optional{% endif %}
- RAM: {{ model_type.memory_requirements }}MB minimum
- Disk Space: {{ model_type.model_size }}MB

## Usage

Basic usage:
```bash
lilypad run {{ model_info.id }} -i INPUT="your input here"
```

{% if usage_examples %}
### Examples

{% for example in usage_examples %}
{{ example.description }}:
```bash
{{ example.command }}
```
{% endfor %}
{% endif %}

## Configuration

Available configuration options:
{% for key, value in model_type.config.items() %}
- `{{ key }}`: {{ value.description }} (default: `{{ value.default }}`)
{% endfor %}

## Output Format

The module produces outputs in the following format:
```json
{{ model_type.output_format | tojson(indent=2) }}
```

## License

This module wraps the {{ model_info.name }} model which is licensed under {{ model_info.license }}.
''',
        
        "test_script.sh.jinja2": '''
#!/bin/bash

# Test script for {{ model_type.task }} module

set -e

{% if test_cases %}
# Run test cases
{% for test in test_cases %}
echo "Running test case: {{ test.name }}"
lilypad run . -i {% for key, value in test.inputs.items() %}{{ key }}="{{ value }}" {% endfor %}

# Verify outputs
echo "Verifying outputs..."
{% for check in test.checks %}
{{ check }}
{% endfor %}

{% endfor %}
{% else %}
# Default test case
echo "Running default test..."
lilypad run . -i INPUT="Test input"
{% endif %}

echo "All tests completed successfully!"
'''
    }
    
    # Write template files
    for filename, content in templates.items():
        file_path = os.path.join(template_dir, filename)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(content.lstrip())

def get_task_specific_templates(task: str) -> Dict[str, str]:
    """Get additional templates specific to a task
    
    Args:
        task: Task type (e.g., text-generation)
        
    Returns:
        Dict of template names and contents
    """
    templates = {
        "text-generation": {
            "inference.py.jinja2": '''
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def main():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForCausalLM.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Get input text
    input_text = os.getenv("INPUT", "Default input text")

    # Generate text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens={{ config.max_new_tokens }},
        do_sample={{ config.do_sample }},
        temperature={{ config.temperature }},
        top_p={{ config.top_p }},
    )

    # Decode and save output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    output = {
        "input": input_text,
        "output": generated_text
    }
    
    # Save results
    with open("/outputs/results.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
'''
        }
    }
    
    return templates.get(task, {})