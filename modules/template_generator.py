"""
Dynamic template generator for Lilypad modules based on model analysis.
"""

import os
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader
from .analyzer import ModelAnalysis

class TemplateGenerator:
    """Generates module files based on model analysis"""
    
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.base_templates = {
            'dockerfile': 'Dockerfile.jinja2',
            'inference': 'inference.py.jinja2',
            'module_config': 'lilypad_module.json.tmpl.jinja2',
            'requirements': 'requirements.txt.jinja2'
        }
        
    def generate_files(
        self,
        analysis: ModelAnalysis,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate all necessary files for a Lilypad module
        
        Args:
            analysis: Complete model analysis
            output_dir: Directory to write files to
            
        Returns:
            Dict mapping file types to their paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = {}
        
        # Generate each file type
        for file_type, template_name in self.base_templates.items():
            output_path = os.path.join(
                output_dir,
                self._get_output_filename(file_type)
            )
            
            content = self._render_template(
                template_name,
                analysis,
                file_type
            )
            
            with open(output_path, 'w') as f:
                f.write(content)
                
            generated_files[file_type] = output_path
            
        return generated_files
        
    def _get_output_filename(self, file_type: str) -> str:
        """Get appropriate filename for each file type"""
        filenames = {
            'dockerfile': 'Dockerfile',
            'inference': 'run_inference.py',
            'module_config': 'lilypad_module.json.tmpl',
            'requirements': 'requirements.txt'
        }
        return filenames.get(file_type, f"{file_type}.txt")
        
    def _render_template(
        self,
        template_name: str,
        analysis: ModelAnalysis,
        file_type: str
    ) -> str:
        """
        Render a template with model analysis data
        
        Args:
            template_name: Name of template file
            analysis: Model analysis data
            file_type: Type of file being generated
            
        Returns:
            Rendered template content
        """
        # Get template context based on file type
        context = self._get_template_context(analysis, file_type)
        
        # Get appropriate template renderer
        template = self.env.get_template(template_name)
        
        return template.render(**context)
        
    def _get_template_context(
        self,
        analysis: ModelAnalysis,
        file_type: str
    ) -> Dict[str, Any]:
        """Get template context based on file type"""
        # Base context used by all templates
        base_context = {
            'model_id': analysis.model_id,
            'task_type': analysis.task_type,
            'architecture': analysis.architecture,
            'framework': analysis.framework
        }
        
        # File-specific contexts
        contexts = {
            'dockerfile': {
                **base_context,
                'cuda_version': '11.8.0',  # Could be detected from requirements
                'python_version': '3.9',
                'system_packages': [
                    pkg for pkg in analysis.required_packages 
                    if not pkg.startswith(('torch', 'transformers', 'diffusers'))
                ],
                'requirements_file': 'requirements.txt'
            },
            'inference': {
                **base_context,
                'input_types': analysis.input_types,
                'output_types': analysis.output_types,
                'model_loader': analysis.model_loader,
                'processor_type': analysis.processor_type,
                'generation_params': analysis.generation_params,
                'special_tokens': analysis.special_tokens
            },
            'module_config': {
                **base_context,
                'machine': {
                    'gpu': 1 if analysis.hardware_requirements['requires_gpu'] else 0,
                    'cpu': 1000,
                    'ram': 8000
                },
                'timeout': 1800,
                'concurrency': 1
            },
            'requirements': {
                **base_context,
                'packages': analysis.required_packages
            }
        }
        
        return contexts.get(file_type, base_context)
        
    def _generate_error_handling(
        self,
        analysis: ModelAnalysis
    ) -> str:
        """Generate error handling code based on model type"""
        error_handlers = []
        
        # Add input validation
        for input_type in analysis.input_types:
            if input_type == 'text':
                error_handlers.append("""
    if not isinstance(text_input, str):
        raise ValueError("Text input must be a string")
    if len(text_input.strip()) == 0:
        raise ValueError("Text input cannot be empty")
""")
            elif input_type == 'image':
                error_handlers.append("""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")
""")
                
        return '\n'.join(error_handlers)
        
    def _generate_output_handling(
        self,
        analysis: ModelAnalysis
    ) -> str:
        """Generate output handling code based on model type"""
        output_handlers = []
        
        for output_type in analysis.output_types:
            if output_type == 'image':
                output_handlers.append("""
    # Save image in multiple formats
    save_paths = {}
    
    # Save as PNG
    png_path = os.path.join("/outputs", "generated_image.png")
    image.save(png_path, "PNG")
    save_paths["png"] = png_path
    
    # Save as WEBP for web
    webp_path = os.path.join("/outputs", "generated_image.webp")
    image.save(webp_path, "WEBP", quality=90)
    save_paths["webp"] = webp_path
""")
            elif output_type == 'text':
                output_handlers.append("""
    # Format text output
    if isinstance(output, (list, tuple)):
        output = output[0] if output else ""
    output = str(output).strip()
""")
                
        return '\n'.join(output_handlers)