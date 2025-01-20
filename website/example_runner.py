import os
import json
import docker
import tempfile
import shutil
from typing import Dict, Any, Optional

class ModuleRunner:
    """Local test runner for generated modules"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.temp_dirs = []
    
    def run_test(self, 
                 files: Dict[str, str], 
                 input_data: Any, 
                 input_type: str,
                 model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run a test of the generated module
        
        Args:
            files: Dictionary of module files
            input_data: Input data for the model
            input_type: Type of input (text, image, json)
            model_params: Optional model parameters
            
        Returns:
            Dictionary containing results and logs
        """
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            # Write module files
            for filename, content in files.items():
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write(content)
            
            # Create input and output directories
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = os.path.join(temp_dir, 'outputs')
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            
            # Write input file
            input_path = self._write_input(input_dir, input_data, input_type)
            
            # Build Docker image
            image_tag = 'lilypad-test-module'
            self.docker_client.images.build(
                path=temp_dir,
                tag=image_tag,
                rm=True
            )
            
            # Run container
            environment = {
                'INPUT_PATH': f'/workspace/input/{os.path.basename(input_path)}'
            }
            if model_params:
                environment['MODEL_PARAMS'] = json.dumps(model_params)
            
            container = self.docker_client.containers.run(
                image_tag,
                environment=environment,
                volumes={
                    input_dir: {'bind': '/workspace/input', 'mode': 'ro'},
                    output_dir: {'bind': '/outputs', 'mode': 'rw'}
                },
                detach=True
            )
            
            # Wait for container to finish
            result = container.wait()
            logs = container.logs().decode('utf-8')
            
            # Read results
            output_file = os.path.join(output_dir, 'result.json')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
            else:
                results = {'error': 'No output generated'}
            
            # Cleanup
            container.remove()
            self.docker_client.images.remove(image_tag, force=True)
            
            return {
                'results': results,
                'logs': logs,
                'exit_code': result['StatusCode']
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'logs': '',
                'exit_code': 1
            }
        
        finally:
            # Cleanup temporary directories
            for temp_dir in self.temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            self.temp_dirs = []
    
    def _write_input(self, input_dir: str, input_data: Any, input_type: str) -> str:
        """Write input data to appropriate file"""
        if input_type == 'text':
            input_path = os.path.join(input_dir, 'input.txt')
            with open(input_path, 'w') as f:
                f.write(str(input_data))
                
        elif input_type == 'json':
            input_path = os.path.join(input_dir, 'input.json')
            with open(input_path, 'w') as f:
                json.dump(input_data, f)
                
        elif input_type == 'image':
            input_path = os.path.join(input_dir, 'input.jpg')
            # Assuming input_data is base64 encoded image
            import base64
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(input_data))
                
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
            
        return input_path