"""
Tests for text processing handler
"""

import pytest
import os
from modules.handlers.text import TextHandler
from modules.types.model_types import ModelType
from modules.types.task_types import TaskType, TaskRequirements
from ..utils import MockModel, MockProcessor, assert_valid_output

@pytest.fixture
def text_handler():
    """Create a text handler for testing"""
    return TextHandler(
        model_id="test/text-model",
        task="text-classification"
    )

@pytest.fixture
def mock_model():
    """Create a mock model"""
    return MockModel("text-classification")

@pytest.fixture
def mock_processor():
    """Create a mock processor"""
    return MockProcessor("text-classification")

def test_text_handler_initialization():
    """Test text handler initialization"""
    handler = TextHandler("test/model", "text-classification")
    assert handler.model_id == "test/model"
    assert handler.task == "text-classification"

def test_text_handler_imports():
    """Test import statement generation"""
    handler = TextHandler("test/model", "text-classification")
    imports = handler.generate_imports()
    
    # Check for required imports
    assert "from transformers import" in imports
    assert "AutoTokenizer" in imports
    assert "AutoModelForSequenceClassification" in imports

def test_text_classification_inference(text_handler, mock_model, mock_processor):
    """Test text classification inference"""
    # Generate and evaluate inference code
    inference_code = text_handler.generate_inference()
    
    # Create namespace and execute code
    namespace = {
        'torch': pytest.importorskip("torch"),
        'os': os,
        'model': mock_model,
        'tokenizer': mock_processor
    }
    exec(inference_code, namespace)
    
    # Run inference
    result = namespace['process_input']("test text", mock_model, mock_processor)
    
    # Validate output
    assert_valid_output(result, [
        'predictions',
        'text',
        'metadata'
    ])
    
    # Check predictions format
    assert isinstance(result['predictions'], list)
    for pred in result['predictions']:
        assert 'label' in pred
        assert 'confidence' in pred
        assert isinstance(pred['confidence'], float)
        assert 0 <= pred['confidence'] <= 1

def test_text_generation_inference():
    """Test text generation inference"""
    handler = TextHandler("test/model", "text-generation")
    inference_code = handler.generate_inference()
    
    # Create namespace and execute code
    namespace = {
        'torch': pytest.importorskip("torch"),
        'os': os,
        'model': MockModel("text-generation"),
        'tokenizer': MockProcessor("text-generation")
    }
    exec(inference_code, namespace)
    
    # Run inference
    result = namespace['process_input']("test prompt", namespace['model'], namespace['tokenizer'])
    
    # Validate output
    assert_valid_output(result, [
        'generated_texts',
        'prompt',
        'parameters',
        'metadata'
    ])
    
    assert isinstance(result['generated_texts'], list)
    assert result['prompt'] == "test prompt"
    assert 'temperature' in result['parameters']

def test_handler_requirements():
    """Test handler requirements generation"""
    handler = TextHandler("test/model", "text-classification")
    reqs = handler.get_requirements()
    
    # Check for essential requirements
    assert any('transformers' in req for req in reqs)
    assert any('torch' in req for req in reqs)
    assert any('sentencepiece' in req for req in reqs)

def test_gpu_requirements():
    """Test GPU requirement detection"""
    # Models that should require GPU
    gpu_models = [
        ("gpt2-large", "text-generation"),
        ("t5-large", "translation"),
        ("facebook/opt-1.3b", "text-generation")
    ]
    
    for model_id, task in gpu_models:
        handler = TextHandler(model_id, task)
        assert handler.requires_gpu(), f"{model_id} should require GPU"
    
    # Models that might not require GPU
    cpu_models = [
        ("prajjwal1/bert-tiny", "text-classification"),
        ("distilbert-base-uncased", "text-classification")
    ]
    
    for model_id, task in cpu_models:
        handler = TextHandler(model_id, task)
        assert not handler.requires_gpu(), f"{model_id} should not require GPU"

def test_error_handling(text_handler, mock_model, mock_processor):
    """Test error handling in inference code"""
    inference_code = text_handler.generate_inference()
    
    # Create namespace and execute code
    namespace = {
        'torch': pytest.importorskip("torch"),
        'os': os,
        'model': mock_model,
        'tokenizer': mock_processor
    }
    exec(inference_code, namespace)
    
    # Test with invalid input
    result = namespace['process_input'](None, mock_model, mock_processor)
    assert 'error' in result
    
    # Test with model exception
    class BrokenModel(MockModel):
        def __call__(self, **kwargs):
            raise RuntimeError("Model error")
    
    result = namespace['process_input']("test", BrokenModel("text-classification"), mock_processor)
    assert 'error' in result