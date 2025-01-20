import yaml
import docker
import os
from typing import Dict, List, Optional

class ConfigValidator:
    def __init__(self):
        self.docker_client = docker.from_env()
        
    def validate_dockerfile(self, dockerfile_content: str) -> List[str]:
        """Validate Dockerfile syntax and dependencies"""
        errors = []
        
        # Create temporary Dockerfile
        with open('temp_Dockerfile', 'w') as f:
            f.write(dockerfile_content)
            
        try:
            # Try to build the image
            self.docker_client.images.build(
                path='.',
                dockerfile='temp_Dockerfile',
                tag='temp_validation'
            )
        except Exception as e:
            errors.append(f"Dockerfile validation error: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists('temp_Dockerfile'):
                os.remove('temp_Dockerfile')
                
        return errors
        
    def validate_yaml(self, yaml_content: str) -> List[str]:
        """Validate module.yaml syntax and required fields"""
        errors = []
        
        try:
            config = yaml.safe_load(yaml_content)
            
            # Check required fields
            required_fields = ['name', 'version', 'description', 'input', 'output']
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field: {field}")
                    
            # Validate input configuration
            if 'input' in config:
                for input_config in config['input']:
                    if 'name' not in input_config or 'type' not in input_config:
                        errors.append("Input configuration must include 'name' and 'type'")
                        
            # Validate output configuration
            if 'output' in config:
                for output_config in config['output']:
                    if 'name' not in output_config or 'type' not in output_config:
                        errors.append("Output configuration must include 'name' and 'type'")
                    if output_config.get('type') == 'file' and 'path' not in output_config:
                        errors.append("File outputs must specify 'path'")
                        
        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {str(e)}")
            
        return errors
        
    def validate_python_script(self, script_content: str) -> List[str]:
        """Validate Python script syntax and required functions"""
        errors = []
        
        try:
            # Check syntax
            compile(script_content, '<string>', 'exec')
            
            # Check for required elements
            required_elements = [
                ('if __name__ == "__main__":', "Missing main guard"),
                ('def main():', "Missing main function"),
                ('/outputs', "Missing output directory reference"),
                ('json.dump', "Missing JSON output generation")
            ]
            
            for element, error_msg in required_elements:
                if element not in script_content:
                    errors.append(error_msg)
                    
        except SyntaxError as e:
            errors.append(f"Python syntax error: {str(e)}")
            
        return errors
        
    def validate_requirements(self, requirements_content: str) -> List[str]:
        """Validate requirements.txt format and package names"""
        errors = []
        
        required_packages = ['transformers', 'torch']
        found_packages = []
        
        for line in requirements_content.split('\n'):
            if line.strip() and not line.startswith('#'):
                package = line.split('==')[0].strip()
                found_packages.append(package)
                
                # Check package name format
                if not package.replace('-', '').replace('_', '').isalnum():
                    errors.append(f"Invalid package name format: {package}")
                    
        # Check required packages
        for package in required_packages:
            if package not in found_packages:
                errors.append(f"Missing required package: {package}")
                
        return errors
        
    def validate_all(self, files: Dict[str, str]) -> Dict[str, List[str]]:
        """Validate all configuration files"""
        validation_results = {}
        
        if 'Dockerfile' in files:
            validation_results['Dockerfile'] = self.validate_dockerfile(files['Dockerfile'])
            
        if 'module.yaml' in files:
            validation_results['module.yaml'] = self.validate_yaml(files['module.yaml'])
            
        if 'run_inference.py' in files:
            validation_results['run_inference.py'] = self.validate_python_script(files['run_inference.py'])
            
        if 'requirements.txt' in files:
            validation_results['requirements.txt'] = self.validate_requirements(files['requirements.txt'])
            
        return validation_results