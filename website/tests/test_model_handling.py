import unittest
import os
import json
import requests
from unittest.mock import Mock, patch
from example_runner import ModuleRunner
from model_templates import ModelTemplate

class TestModelHandling(unittest.TestCase):
    """Test model loading and inference"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_models = {
            'resnet': 'microsoft/resnet-50',
            'vit': 'google/vit-base-patch16-224',
            'deit': 'facebook/deit-base-patch16-224',
            'convnext': 'facebook/convnext-tiny-224'
        }
        cls.runner = ModuleRunner()
        
    def setUp(self):
        """Set up before each test"""
        self.model_info_mock = Mock()
        self.model_info_mock.id = 'test/model'
        self.model_info_mock.size_in_bytes = 100000000
        self.model_info_mock.tags = ['image-classification']
        
    def test_resnet_loading(self):
        """Test ResNet model loading and basic inference"""
        model_url = self.test_models['resnet']
        template = ModelTemplate(self.model_info_mock)
        files = {
            'Dockerfile': template.generate_dockerfile(),
            'requirements.txt': template.generate_requirements(),
            'run_inference.py': template.generate_run_inference()
        }
        
        # Test with a simple image
        with open('tests/test_images/sample_image.jpg', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)
        self.assertIn('predictions', result['results'])
        
    def test_model_error_handling(self):
        """Test error handling for model loading issues"""
        template = ModelTemplate(self.model_info_mock)
        files = {
            'Dockerfile': template.generate_dockerfile(),
            'requirements.txt': template.generate_requirements(),
            'run_inference.py': template.generate_run_inference()
        }
        
        # Test with non-existent model
        with patch('transformers.AutoModelForImageClassification.from_pretrained') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            
            result = self.runner.run_test(
                files=files,
                input_data="test_data",
                input_type='text'
            )
            
            self.assertNotEqual(result['exit_code'], 0)
            self.assertIn('error', result['results'])
            
    def test_model_parameter_handling(self):
        """Test handling of model parameters"""
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        # Test with custom parameters
        params = {
            'top_k': 3,
            'threshold': 0.5
        }
        
        result = self.runner.run_test(
            files=files,
            input_data="test_data",
            input_type='text',
            model_params=params
        )
        
        self.assertEqual(result['exit_code'], 0)
        
    def test_model_resource_requirements(self):
        """Test resource requirement estimation"""
        # Test small model
        self.model_info_mock.size_in_bytes = 50 * 1024 * 1024  # 50MB
        template = ModelTemplate(self.model_info_mock)
        self.assertEqual(template._estimate_resources()['memory'], '2Gi')
        
        # Test large model
        self.model_info_mock.size_in_bytes = 1.5 * 1024 * 1024 * 1024  # 1.5GB
        template = ModelTemplate(self.model_info_mock)
        self.assertEqual(template._estimate_resources()['memory'], '8Gi')
        
if __name__ == '__main__':
    unittest.main()