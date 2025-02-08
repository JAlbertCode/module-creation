"""
Tests for model analyzer functionality
"""

import pytest
from modules.analyzer import ModelAnalyzer, ModelAnalysis

def test_model_analysis_text_generation(mock_model_info):
    """Test analysis of text generation model"""
    analyzer = ModelAnalyzer()
    model_info = mock_model_info("gpt2", "text-generation", "GPT2LMHeadModel")
    
    analysis = analyzer._analyze_from_info(model_info)
    
    assert isinstance(analysis, ModelAnalysis)
    assert analysis.task_type == "text-generation"
    assert analysis.framework == "pytorch"
    assert "transformers" in analysis.required_packages
    assert analysis.model_loader == "AutoModelForCausalLM"
    assert analysis.processor_type == "AutoTokenizer"
    assert analysis.input_types == ["text"]
    assert analysis.output_types == ["text"]

def test_model_analysis_stable_diffusion(mock_model_info):
    """Test analysis of Stable Diffusion model"""
    analyzer = ModelAnalyzer()
    model_info = mock_model_info(
        "CompVis/stable-diffusion-v1-4",
        "text-to-image",
        "StableDiffusionPipeline"
    )
    
    analysis = analyzer._analyze_from_info(model_info)
    
    assert analysis.task_type == "text-to-image"
    assert "diffusers" in analysis.required_packages
    assert analysis.model_loader == "StableDiffusionPipeline"
    assert analysis.processor_type is None
    assert analysis.input_types == ["text"]
    assert analysis.output_types == ["image"]
    assert analysis.hardware_requirements["requires_gpu"] is True

def test_hardware_requirements_detection(mock_model_info):
    """Test hardware requirements detection"""
    analyzer = ModelAnalyzer()
    
    # Large model
    large_model = mock_model_info("large-model", "text-generation", "LargeModel")
    large_model["size_in_bytes"] = 15 * 1024 * 1024 * 1024  # 15GB
    
    analysis = analyzer._analyze_from_info(large_model)
    assert analysis.hardware_requirements["requires_gpu"] is True
    assert analysis.hardware_requirements["minimum_gpu_memory"] == "24GB"
    
    # Small model
    small_model = mock_model_info("small-model", "text-generation", "SmallModel")
    small_model["size_in_bytes"] = 500 * 1024 * 1024  # 500MB
    
    analysis = analyzer._analyze_from_info(small_model)
    assert analysis.hardware_requirements["requires_gpu"] is False

def test_generation_params_detection(mock_model_info):
    """Test generation parameters detection"""
    analyzer = ModelAnalyzer()
    model_info = mock_model_info("gpt2", "text-generation", "GPT2LMHeadModel")
    
    # Add generation params to config
    model_info["config"]["max_length"] = 128
    model_info["config"]["do_sample"] = True
    model_info["config"]["temperature"] = 0.7
    
    analysis = analyzer._analyze_from_info(model_info)
    
    assert "max_length" in analysis.generation_params
    assert analysis.generation_params["max_length"] == 128
    assert analysis.generation_params["do_sample"] is True
    assert analysis.generation_params["temperature"] == 0.7

def test_special_tokens_detection(mock_model_info):
    """Test special tokens detection"""
    analyzer = ModelAnalyzer()
    model_info = mock_model_info("gpt2", "text-generation", "GPT2LMHeadModel")
    
    # Add special tokens to config
    model_info["config"]["bos_token_id"] = 1
    model_info["config"]["eos_token_id"] = 2
    model_info["config"]["pad_token_id"] = 0
    
    analysis = analyzer._analyze_from_info(model_info)
    
    assert analysis.special_tokens["bos_token_id"] == 1
    assert analysis.special_tokens["eos_token_id"] == 2
    assert analysis.special_tokens["pad_token_id"] == 0

def test_error_handling():
    """Test error handling in analyzer"""
    analyzer = ModelAnalyzer()
    
    # Invalid model ID
    with pytest.raises(ValueError):
        analyzer.analyze_model("nonexistent/model")
    
    # Missing required info
    with pytest.raises(ValueError):
        analyzer._analyze_from_info({})
        
    # Invalid task type
    info = {"pipeline_tag": "invalid_task"}
    with pytest.raises(ValueError):
        analyzer._analyze_from_info(info)