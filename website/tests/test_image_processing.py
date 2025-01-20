import unittest
import os
import base64
from PIL import Image
import io
from example_runner import ModuleRunner
from model_templates import ModelTemplate

class TestImageProcessing(unittest.TestCase):
    """Test image processing and handling"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.setup_test_images()
        cls.runner = ModuleRunner()
        
    @classmethod
    def setup_test_images(cls):
        """Create test images in different formats"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        
        # Save in different formats
        test_images_dir = 'tests/test_images'
        os.makedirs(test_images_dir, exist_ok=True)
        
        # JPEG
        img.save(os.path.join(test_images_dir, 'sample_image.jpg'), 'JPEG')
        
        # PNG
        img.save(os.path.join(test_images_dir, 'sample_image.png'), 'PNG')
        
        # WebP
        img.save(os.path.join(test_images_dir, 'sample_image.webp'), 'WebP')
        
        # Large image
        large_img = Image.new('RGB', (2048, 2048), color='blue')
        large_img.save(os.path.join(test_images_dir, 'large_image.jpg'), 'JPEG')
        
        # Grayscale
        gray_img = Image.new('L', (224, 224), color=128)
        gray_img.save(os.path.join(test_images_dir, 'grayscale_image.png'), 'PNG')
        
        # Create corrupt image
        with open(os.path.join(test_images_dir, 'corrupt_image.jpg'), 'wb') as f:
            f.write(b'This is not a valid image file')
    
    def setUp(self):
        """Set up before each test"""
        self.model_info_mock = Mock()
        self.model_info_mock.id = 'test/model'
        self.model_info_mock.tags = ['image-classification']
    
    def test_jpeg_support(self):
        """Test JPEG image processing"""
        with open('tests/test_images/sample_image.jpg', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)
    
    def test_png_support(self):
        """Test PNG image processing"""
        with open('tests/test_images/sample_image.png', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)
    
    def test_webp_support(self):
        """Test WebP image processing"""
        with open('tests/test_images/sample_image.webp', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)
    
    def test_large_image_handling(self):
        """Test handling of large images"""
        with open('tests/test_images/large_image.jpg', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)
        
    def test_corrupt_image_handling(self):
        """Test handling of corrupt images"""
        with open('tests/test_images/corrupt_image.jpg', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertNotEqual(result['exit_code'], 0)
        self.assertIn('error', result['results'])
    
    def test_grayscale_conversion(self):
        """Test handling of grayscale images"""
        with open('tests/test_images/grayscale_image.png', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        template = ModelTemplate(self.model_info_mock)
        files = template.generate_all_files()
        
        result = self.runner.run_test(
            files=files,
            input_data=image_data,
            input_type='image'
        )
        
        self.assertEqual(result['exit_code'], 0)

if __name__ == '__main__':
    unittest.main()