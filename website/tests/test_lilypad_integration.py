import unittest
import os
import json
import subprocess
import tempfile
import time
from pathlib import Path

class TestLilypadIntegration(unittest.TestCase):
    """Test integration with Lilypad network"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Check if Lilypad CLI is installed
        try:
            subprocess.run(['lilypad', '--version'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise unittest.SkipTest("Lilypad CLI not installed")
        except FileNotFoundError:
            raise unittest.SkipTest("Lilypad CLI not found")
            
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
    def setUp(self):
        """Set up before each test"""
        # Create test image
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        self._create_test_image()
        
        # Set up model configuration
        self.model_id = "microsoft/resnet-50"
        
    def _create_test_image(self):
        """Create a test image for inference"""
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(self.test_image_path)
        
    def _deploy_to_lilypad(self):
        """Deploy the module to Lilypad"""
        deploy_cmd = ['lilypad', 'module', 'deploy', '.']
        result = subprocess.run(deploy_cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Deployment failed: {result.stderr}")
        return result.stdout
        
    def _run_on_lilypad(self):
        """Run the deployed module on Lilypad"""
        run_cmd = [
            'lilypad', 'run',
            '--input', f'INPUT_PATH={self.test_image_path}',
            '--input', f'MODEL_ID={self.model_id}',
            'image-classification'
        ]
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Run failed: {result.stderr}")
        return result.stdout
        
    def _get_job_status(self, job_id):
        """Get status of a Lilypad job"""
        status_cmd = ['lilypad', 'status', job_id]
        result = subprocess.run(status_cmd, capture_output=True, text=True)
        return result.stdout
        
    def _get_job_results(self, job_id):
        """Get results of a completed job"""
        results_cmd = ['lilypad', 'get-results', job_id]
        result = subprocess.run(results_cmd, capture_output=True, text=True)
        return result.stdout
        
    def test_module_deployment(self):
        """Test deploying module to Lilypad"""
        deployment_output = self._deploy_to_lilypad()
        self.assertIn('successfully deployed', deployment_output.lower())
        
    def test_module_execution(self):
        """Test running module on Lilypad"""
        # Deploy first
        self._deploy_to_lilypad()
        
        # Run the module
        run_output = self._run_on_lilypad()
        job_id = run_output.strip()
        
        # Wait for completion (with timeout)
        timeout = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self._get_job_status(job_id)
            if 'completed' in status.lower():
                break
            time.sleep(10)
        else:
            self.fail("Job timed out")
            
        # Get and verify results
        results = self._get_job_results(job_id)
        try:
            result_data = json.loads(results)
            self.assertIn('predictions', result_data)
            self.assertEqual(result_data['status'], 'success')
        except json.JSONDecodeError:
            self.fail("Invalid JSON response")
            
    def test_error_handling(self):
        """Test error handling on Lilypad network"""
        # Test with invalid model ID
        self.model_id = "invalid/model"
        run_output = self._run_on_lilypad()
        job_id = run_output.strip()
        
        # Wait for completion
        timeout = 60
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self._get_job_status(job_id)
            if 'failed' in status.lower():
                break
            time.sleep(5)
            
        results = self._get_job_results(job_id)
        result_data = json.loads(results)
        self.assertEqual(result_data['status'], 'error')
        
    def test_resource_limits(self):
        """Test module behavior with resource constraints"""
        # TODO: Implement resource limit testing
        pass
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove temporary directory
        import shutil
        shutil.rmtree(cls.temp_dir)
        
if __name__ == '__main__':
    unittest.main()