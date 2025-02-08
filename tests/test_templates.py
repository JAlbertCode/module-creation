"""
Tests for template generation and validation
"""

import os
import json
import pytest
from modules.template_generator import TemplateGenerator
from modules.analyzer import ModelAnalysis

@pytest.fixture
def sample_analysis():
    """Sample model analysis for testing"""
    return ModelAnalysis(
        model_id="test/model",
        task_type="text-generation",
        architecture="TestModel",
        framework="pytorch",
        pipeline_type="text-generation",
        input_types=["text"],
        output_types=["text"],
        required_packages=["torch", "transformers"],
        model_params={"hidden_size": 768},
        generation_params={
            "max_length": 128,
            "temperature": 0.7
        },
        special_tokens={
            "pad_token_id": 0,
            "eos_token_id": 2
        },
        hardware_requirements={
            "requires_gpu": True,
            "minimum_gpu_memory": "8GB"
        },
        model_loader="AutoModelForCausalLM",
        processor_type="AutoTokenizer"
    )

def test_dockerfile_generation(temp_output_dir, sample_analysis):
    """Test Dockerfile template generation"""
    generator = TemplateGenerator()
    
    # Generate Dockerfile
    dockerfile_path = os.path.join(temp_output_dir, "Dockerfile")
    content = generator._render_template(
        "Dockerfile.jinja2",
        sample_analysis,
        "dockerfile"
    )
    
    # Write content for inspection
    with open(dockerfile_path, "w") as f:
        f.write(content)
    
    # Validate content
    with open(dockerfile_path, "r") as f:
        dockerfile = f.read()
        
        # Check required components
        assert "FROM nvidia/cuda:" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "COPY requirements.txt" in dockerfile
        assert "RUN pip install" in dockerfile
        assert "COPY run_inference.py" in dockerfile
        
        # Check environment setup
        assert "ENV PYTHONUNBUFFERED=1" in dockerfile
        assert "ENV TRANSFORMERS_OFFLINE=1" in dockerfile
        
        # Check model setup
        assert "RUN mkdir -p /inputs /outputs /model" in dockerfile

def test_inference_script_generation(temp_output_dir, sample_analysis):
    """Test inference script template generation"""
    generator = TemplateGenerator()
    
    # Generate inference script
    script_path = os.path.join(temp_output_dir, "run_inference.py")
    content = generator._render_template(
        "inference.py.jinja2",
        sample_analysis,
        "inference"
    )
    
    # Write content for inspection
    with open(script_path, "w") as f:
        f.write(content)
    
    # Validate content
    with open(script_path, "r") as f:
        script = f.read()
        
        # Check imports
        assert "import torch" in script
        assert "from transformers import" in script
        
        # Check main components
        assert "def load_model():" in script
        assert "def main():" in script
        assert "if __name__ == \"__main__\":" in script
        
        # Check model loading
        assert sample_analysis.model_loader in script
        assert sample_analysis.processor_type in script
        
        # Check error handling
        assert "try:" in script
        assert "except Exception as e:" in script

def test_module_config_generation(temp_output_dir, sample_analysis):
    """Test Lilypad module config template generation"""
    generator = TemplateGenerator()
    
    # Generate config
    config_path = os.path.join(temp_output_dir, "lilypad_module.json.tmpl")
    content = generator._render_template(
        "lilypad_module.json.tmpl.jinja2",
        sample_analysis,
        "module_config"
    )
    
    # Write content for inspection
    with open(config_path, "w") as f:
        f.write(content)
    
    # Validate content
    with open(config_path, "r") as f:
        config = f.read()
        
        # Verify it's valid JSON template
        assert "machine" in config
        assert "\"gpu\":" in config
        assert "\"ram\":" in config
        
        # Check Docker configuration
        assert "\"Docker\": {" in config
        assert "\"Image\":" in config
        assert "\"EnvironmentVariables\":" in config
        
        # Check expected variables
        assert "TEXT_INPUT" in config
        assert "max_length" in config.lower()
        assert "temperature" in config.lower()

def test_requirements_generation(temp_output_dir, sample_analysis):
    """Test requirements.txt template generation"""
    generator = TemplateGenerator()
    
    # Generate requirements
    reqs_path = os.path.join(temp_output_dir, "requirements.txt")
    content = generator._render_template(
        "requirements.txt.jinja2",
        sample_analysis,
        "requirements"
    )
    
    # Write content for inspection
    with open(reqs_path, "w") as f:
        f.write(content)
    
    # Validate content
    with open(reqs_path, "r") as f:
        requirements = f.read()
        
        # Check required packages
        for package in sample_analysis.required_packages:
            assert package in requirements

def test_template_variables(temp_output_dir, sample_analysis):
    """Test template variable handling"""
    generator = TemplateGenerator()
    
    # Test with missing optional fields
    minimal_analysis = ModelAnalysis(
        model_id="test/model",
        task_type="text-generation",
        architecture="TestModel",
        framework="pytorch",
        pipeline_type="text-generation",
        input_types=["text"],
        output_types=["text"],
        required_packages=["torch"],
        model_params={},
        generation_params={},
        special_tokens={},
        hardware_requirements={
            "requires_gpu": False
        },
        model_loader="AutoModel"
    )
    
    # Should not raise errors
    for template_name in generator.base_templates.values():
        content = generator._render_template(
            template_name,
            minimal_analysis,
            template_name.split('.')[0]
        )
        assert content is not None

def test_error_handling(temp_output_dir, sample_analysis):
    """Test template error handling"""
    generator = TemplateGenerator()
    
    # Test with invalid template
    with pytest.raises(Exception):
        generator._render_template(
            "nonexistent.jinja2",
            sample_analysis,
            "invalid"
        )
    
    # Test with invalid file type
    with pytest.raises(KeyError):
        generator._render_template(
            "Dockerfile.jinja2",
            sample_analysis,
            "invalid_type"
        )

def test_performance(temp_output_dir, sample_analysis):
    """Test template generation performance"""
    generator = TemplateGenerator()
    
    # Generate all templates multiple times
    for _ in range(10):
        for template_name in generator.base_templates.values():
            content = generator._render_template(
                template_name,
                sample_analysis,
                template_name.split('.')[0]
            )
            assert content is not None
            
    # TODO: Add actual timing measurements and thresholds