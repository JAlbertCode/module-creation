"""Utility functions for generating module files"""

from . import handlers

def generate_dockerfile(model_type):
    """Generate Dockerfile with appropriate dependencies"""
    system_packages = handlers.get_system_packages(model_type['input'])
    system_packages = ' '.join(system_packages)

    return f'''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    {system_packages} \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /outputs /workspace/input

# Copy inference script
COPY run_inference.py .

# Set entrypoint
ENTRYPOINT ["python", "/workspace/run_inference.py"]'''

def generate_requirements(model_type):
    """Generate requirements.txt content"""
    base_reqs = [
        "transformers==4.36.0",
        "torch==2.1.0"
    ]
    
    extra_reqs = handlers.get_requirements(model_type['input'])
    
    return "\\n".join(base_reqs + extra_reqs)