"""
Tests for model conversion functionality
"""

import os
import pytest
from modules.converter import HFToLilypadConverter

def test_model_conversion(test_models, temp_output_dir, validation_checks):
    """Test full model conversion process"""
    converter = HFToLilypadConverter()
    
    for model_type, model_info in test_models.items():
        # Convert model
        result = converter.convert_model(
            model_id=model_info["model_id"],
            output_dir=temp_output_dir
        )
        
        # Validate conversion result
        assert result.model_id == model_info["model_id"]
        assert os.path.exists(result.module_path)
        
        # Check generated files
        validation_results = validation_checks(
            result.module_path,
            expected_files={
                "Dockerfile": [
                    "FROM nvidia/cuda:",
                    "WORKDIR /app",
                    "COPY requirements.txt ."
                ],
                "run_inference.py": [
                    "import torch",
                    "def main():",
                    "if __name__ == \"__main__\":"
                ]
            }
        )
        
        for filename, checks in validation_results.items():
            assert checks["exists"], f"Missing file: {filename}"
            assert all(checks["content_checks"]), f"Invalid content in {filename}"

def test_custom_configuration(test_models, temp_output_dir):
    """Test conversion with custom configuration"""
    converter = HFToLilypadConverter()
    
    custom_config = {
        "machine": {
            "gpu": 2,
            "cpu": 2000,
            "ram": 16000
        },
        "generation_params": {
            "max_length": 200,
            "temperature": 0.8
        }
    }
    
    result = converter.convert_model(
        model_id=test_models["text_generation"]["model_id"],
        output_dir=temp_output_dir,
        custom_config=custom_config
    )
    
    # Check if custom config was applied
    config_path = os.path.join(result.module_path, "lilypad_module.json.tmpl")
    assert os.path.exists(config_path)
    
    with open(config_path, 'r') as f:
        content = f.read()
        assert '"gpu": 2' in content
        assert '"cpu": 2000' in content
        assert '"ram": 16000' in content

def test_error_handling(temp_output_dir):
    """Test error handling in converter"""
    converter = HFToLilypadConverter()
    
    # Invalid model ID
    with pytest.raises(Exception) as exc_info:
        converter.convert_model(
            model_id="nonexistent/model",
            output_dir=temp_output_dir
        )
    assert "Failed to convert model" in str(exc_info.value)
    
    # Invalid output directory
    with pytest.raises(Exception):
        converter.convert_model(
            model_id="gpt2",
            output_dir="/nonexistent/path"
        )
        
    # Invalid custom config
    with pytest.raises(ValueError):
        converter.convert_model(
            model_id="gpt2",
            output_dir=temp_output_dir,
            custom_config={"invalid": "config"}
        )

def test_validation(test_models, temp_output_dir):
    """Test module validation"""
    converter = HFToLilypadConverter()
    
    # Convert and validate a model
    result = converter.convert_model(
        model_id=test_models["text_generation"]["model_id"],
        output_dir=temp_output_dir
    )
    
    # Should pass validation
    assert converter.validate_module(result.module_path)
    
    # Test with missing files
    os.remove(os.path.join(result.module_path, "Dockerfile"))
    with pytest.raises(ValueError) as exc_info:
        converter.validate_module(result.module_path)
    assert "Missing required file" in str(exc_info.value)