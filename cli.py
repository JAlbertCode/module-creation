#!/usr/bin/env python
"""Command-line interface for Hugging Face to Lilypad module converter"""

import os
import sys
import click
import logging
import json
from typing import Optional
from pathlib import Path
from huggingface_hub import HfApi

from modules.types import model_types
from modules.utils import (
    validation,
    download,
    templates,
    config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Hugging Face to Lilypad module converter

    Create Lilypad modules from Hugging Face models.
    """
    pass

@cli.command()
@click.argument('model_id')
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output directory for generated module'
)
@click.option(
    '--use-safetensors/--no-safetensors',
    default=True,
    help='Use safetensors format for model weights'
)
@click.option(
    '--check-only',
    is_flag=True,
    help='Only check model compatibility without generating files'
)
def convert(
    model_id: str,
    output: Optional[str] = None,
    use_safetensors: bool = True,
    check_only: bool = False
):
    """Convert a Hugging Face model to a Lilypad module.
    
    MODEL_ID should be the Hugging Face model ID (e.g., bert-base-uncased)
    """
    try:
        # Get model info
        logger.info(f"Getting model info for {model_id}")
        api = HfApi()
        model_info = api.model_info(model_id)
        
        # Detect model type
        model_type = model_types.detect_model_type(model_info)
        logger.info(f"Detected model type: {model_type.task}")
        
        # Check compatibility
        validation_result = validation.check_model_compatibility(model_info)
        if not validation_result.is_valid:
            click.echo(validation.format_validation_message(validation_result))
            sys.exit(1)
            
        if check_only:
            click.echo(validation.format_validation_message(validation_result))
            return
            
        # Set up output directory
        if output is None:
            output = f"lilypad-{model_id.split('/')[-1]}"
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model if needed
        if not check_only:
            logger.info("Downloading model files")
            downloader = download.DownloadManager(
                cache_dir=str(output_dir / "model"),
                use_safetensors=use_safetensors
            )
            
            # Get required files
            allow_patterns = downloader.get_model_files(model_info)
            ignore_patterns = downloader.get_ignore_patterns()
            
            # Download with progress tracking
            model_path = downloader.download_model(
                model_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns
            )
            
            # Verify files
            if not download.verify_downloaded_files(model_path, allow_patterns):
                logger.error("Failed to download all required files")
                sys.exit(1)
        
        # Set up template manager
        template_manager = templates.TemplateManager()
        
        # Generate files
        logger.info("Generating module files")
        
        # Get model configuration
        model_config = model_types.get_default_model_configuration(model_type.task)
        
        # Generate inference script
        inference_code = template_manager.render_inference_script(
            model_type=vars(model_type),
            model_config=model_config
        )
        (output_dir / "run_inference.py").write_text(inference_code)
        
        # Generate Dockerfile
        requirements = model_type.get_requirements()
        dockerfile = template_manager.render_dockerfile(
            model_type=vars(model_type),
            requirements=requirements
        )
        (output_dir / "Dockerfile").write_text(dockerfile)
        
        # Generate module config
        module_config = template_manager.render_module_config(
            model_type=vars(model_type),
            docker_image="<DOCKER_IMAGE>",  # To be filled by user
            env_vars={
                "MODEL_DTYPE": "float16",
                "USE_SAFETENSORS": str(int(use_safetensors))
            }
        )
        (output_dir / "lilypad_module.json.tmpl").write_text(module_config)
        
        # Generate README
        readme = template_manager.render_readme(
            model_type=vars(model_type),
            model_info=vars(model_info)
        )
        (output_dir / "README.md").write_text(readme)
        
        # Generate test script
        test_script = template_manager.render_test_script(
            model_type=vars(model_type)
        )
        (output_dir / "test.sh").write_text(test_script)
        os.chmod(output_dir / "test.sh", 0o755)
        
        logger.info(f"Successfully generated module in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option(
    '--image-name',
    '-n',
    help='Name for the Docker image'
)
def build(directory: str, image_name: Optional[str] = None):
    """Build Docker image for a Lilypad module.
    
    DIRECTORY should be the path to the module directory.
    """
    import subprocess
    
    directory = Path(directory)
    if not (directory / "Dockerfile").exists():
        logger.error(f"No Dockerfile found in {directory}")
        sys.exit(1)
        
    if image_name is None:
        image_name = f"lilypad-{directory.name}:latest"
        
    try:
        logger.info(f"Building Docker image: {image_name}")
        subprocess.run(
            ["docker", "build", "-t", image_name, str(directory)],
            check=True
        )
        logger.info("Successfully built Docker image")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building Docker image: {e}")
        sys.exit(1)

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def test(directory: str):
    """Run tests for a Lilypad module.
    
    DIRECTORY should be the path to the module directory.
    """
    directory = Path(directory)
    test_script = directory / "test.sh"
    
    if not test_script.exists():
        logger.error(f"No test script found in {directory}")
        sys.exit(1)
        
    try:
        logger.info("Running module tests")
        subprocess.run([str(test_script)], check=True)
        logger.info("All tests passed")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    cli()