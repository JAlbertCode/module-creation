"""
Comprehensive model analyzer for automatic configuration generation.
"""

import re
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from huggingface_hub import HfApi, ModelInfo
import requests
from transformers import AutoConfig
from diffusers import DiffusionPipeline

@dataclass
class ModelAnalysis:
    """Complete model analysis results"""
    model_id: str
    task_type: str
    architecture: str
    framework: str
    pipeline_type: str
    input_types: List[str]
    output_types: List[str]
    required_packages: List[str]
    model_params: Dict[str, Any]
    generation_params: Dict[str, Any]
    special_tokens: Dict[str, Any]
    hardware_requirements: Dict[str, Any]
    model_loader: str
    processor_type: Optional[str] = None

class ModelAnalyzer:
    """Analyzer for Hugging Face models"""
    
    FRAMEWORK_PATTERNS = {
        "pytorch": ["torch", "pytorch", "pt"],
        "tensorflow": ["tensorflow", "tf", "keras"],
        "flax": ["flax", "jax"],
    }
    
    def __init__(self):
        self.api = HfApi()
        self.known_architectures = self._load_architecture_mappings()
        
    def _load_architecture_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load known architecture mappings"""
        # This would normally load from a YAML file
        return {
            "gpt": {
                "type": "text-generation",
                "loader": "AutoModelForCausalLM",
                "processor": "AutoTokenizer",
                "packages": ["transformers", "torch"]
            },
            "llama": {
                "type": "text-generation",
                "loader": "AutoModelForCausalLM",
                "processor": "AutoTokenizer",
                "packages": ["transformers", "torch", "accelerate"]
            },
            "stable-diffusion": {
                "type": "text-to-image",
                "loader": "StableDiffusionPipeline",
                "processor": None,
                "packages": ["diffusers", "torch", "transformers"]
            },
            # Add more architecture mappings
        }
        
    def analyze_model(self, model_id: str) -> ModelAnalysis:
        """
        Analyze a model to determine its properties and requirements
        
        Args:
            model_id: Hugging Face model ID (e.g., 'openai/gpt2')
            
        Returns:
            ModelAnalysis object with complete analysis
        """
        # Get model info from Hugging Face
        model_info = self.api.model_info(model_id)
        
        # Parse model card for additional details
        card_data = self._parse_model_card(model_info.cardData)
        
        # Get model config
        config = self._get_model_config(model_id)
        
        # Detect basic properties
        task_type = self._detect_task_type(model_info, config, card_data)
        architecture = self._detect_architecture(model_info, config, card_data)
        framework = self._detect_framework(model_info, card_data)
        
        # Analyze architecture-specific details
        arch_details = self._analyze_architecture(architecture, config)
        
        # Get hardware requirements
        hardware_reqs = self._analyze_hardware_requirements(model_info, config)
        
        # Detect generation parameters
        gen_params = self._detect_generation_params(config, card_data)
        
        return ModelAnalysis(
            model_id=model_id,
            task_type=task_type,
            architecture=architecture,
            framework=framework,
            pipeline_type=arch_details.get("pipeline_type", "default"),
            input_types=arch_details.get("inputs", []),
            output_types=arch_details.get("outputs", []),
            required_packages=self._get_required_packages(arch_details, task_type),
            model_params=config.to_dict() if config else {},
            generation_params=gen_params,
            special_tokens=self._detect_special_tokens(config),
            hardware_requirements=hardware_reqs,
            model_loader=arch_details.get("loader", "AutoModel"),
            processor_type=arch_details.get("processor")
        )
        
    def _parse_model_card(self, card_data: str) -> Dict[str, Any]:
        """Parse model card content for metadata"""
        if not card_data:
            return {}
            
        try:
            # Try to parse as YAML first (common format)
            metadata = yaml.safe_load(card_data)
            if metadata and isinstance(metadata, dict):
                return metadata
        except:
            pass
            
        # Fall back to regex parsing for key information
        metadata = {}
        patterns = {
            "architecture": r"(?:architecture|model type):\s*(\S+)",
            "task": r"(?:task|usage|application):\s*(\S+)",
            "framework": r"(?:framework|library):\s*(\S+)",
            "hardware": r"(?:hardware|requirements):\s*(.+)$",
            "generation": r"(?:generation|inference)[\s\-]params?:\s*(\{[^}]+\}|\[.+\])"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, card_data, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                try:
                    # Try to parse as JSON if it looks like it
                    if value.startswith('{') or value.startswith('['):
                        metadata[key] = json.loads(value)
                    else:
                        metadata[key] = value
                except:
                    metadata[key] = value
                    
        return metadata
        
    def _get_model_config(self, model_id: str) -> Optional[Any]:
        """Get model configuration"""
        try:
            return AutoConfig.from_pretrained(model_id)
        except:
            try:
                # Try diffusers config for stable diffusion models
                return DiffusionPipeline.from_pretrained(model_id, device_map="auto").config
            except:
                return None
                
    def _detect_task_type(
        self,
        model_info: ModelInfo,
        config: Any,
        card_data: Dict[str, Any]
    ) -> str:
        """Detect the model's task type"""
        # Check pipeline tag first
        if model_info.pipeline_tag:
            return model_info.pipeline_tag
            
        # Check model card
        if "task" in card_data:
            return card_data["task"]
            
        # Check config
        if config and hasattr(config, "task"):
            return config.task
            
        # Infer from architecture
        arch = config.architectures[0] if config and hasattr(config, "architectures") else ""
        task_mapping = {
            "ForCausalLM": "text-generation",
            "ForMaskedLM": "fill-mask",
            "ForSequenceClassification": "text-classification",
            "ForQuestionAnswering": "question-answering",
            # Add more mappings
        }
        
        for pattern, task in task_mapping.items():
            if pattern in arch:
                return task
                
        return "unknown"
        
    def _detect_architecture(
        self,
        model_info: ModelInfo,
        config: Any,
        card_data: Dict[str, Any]
    ) -> str:
        """Detect model architecture"""
        # Check config first
        if config and hasattr(config, "architectures"):
            return config.architectures[0]
            
        # Check model card
        if "architecture" in card_data:
            return card_data["architecture"]
            
        # Check model info
        if model_info.config:
            if isinstance(model_info.config, dict):
                return model_info.config.get("architectures", ["Unknown"])[0]
            
        return "Unknown"
        
    def _detect_framework(
        self,
        model_info: ModelInfo,
        card_data: Dict[str, Any]
    ) -> str:
        """Detect the framework used by the model"""
        # Check model info first
        if model_info.library_name:
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(pat in model_info.library_name.lower() for pat in patterns):
                    return framework
                    
        # Check model card
        if "framework" in card_data:
            framework = card_data["framework"].lower()
            for known_fw, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(pat in framework for pat in patterns):
                    return known_fw
                    
        # Default to PyTorch as most common
        return "pytorch"
        
    def _analyze_architecture(
        self,
        architecture: str,
        config: Any
    ) -> Dict[str, Any]:
        """Analyze architecture-specific details"""
        # Check known architectures first
        for arch_pattern, details in self.known_architectures.items():
            if arch_pattern.lower() in architecture.lower():
                return details
                
        # Analyze based on architecture name
        arch_lower = architecture.lower()
        
        if "causallm" in arch_lower or "gpt" in arch_lower:
            return {
                "type": "text-generation",
                "loader": "AutoModelForCausalLM",
                "processor": "AutoTokenizer",
                "inputs": ["text"],
                "outputs": ["text"],
                "packages": ["transformers", "torch"]
            }
            
        if "diffusion" in arch_lower:
            return {
                "type": "text-to-image",
                "loader": "StableDiffusionPipeline",
                "processor": None,
                "inputs": ["text"],
                "outputs": ["image"],
                "packages": ["diffusers", "torch", "transformers"]
            }
            
        # Default generic configuration
        return {
            "type": "unknown",
            "loader": "AutoModel",
            "processor": "AutoProcessor",
            "inputs": ["unknown"],
            "outputs": ["unknown"],
            "packages": ["transformers", "torch"]
        }
        
    def _analyze_hardware_requirements(
        self,
        model_info: ModelInfo,
        config: Any
    ) -> Dict[str, Any]:
        """Analyze hardware requirements"""
        reqs = {
            "requires_gpu": False,
            "minimum_gpu_memory": None,
            "minimum_ram": None
        }
        
        # Check model size
        if model_info.downloads and model_info.downloads > 0:
            size_gb = model_info.downloads / (1024 * 1024 * 1024)  # Convert to GB
            
            if size_gb > 10:
                reqs["requires_gpu"] = True
                reqs["minimum_gpu_memory"] = "24GB"
                reqs["minimum_ram"] = "32GB"
            elif size_gb > 5:
                reqs["requires_gpu"] = True
                reqs["minimum_gpu_memory"] = "16GB"
                reqs["minimum_ram"] = "16GB"
            elif size_gb > 1:
                reqs["requires_gpu"] = True
                reqs["minimum_gpu_memory"] = "8GB"
                reqs["minimum_ram"] = "8GB"
                
        return reqs
        
    def _detect_generation_params(
        self,
        config: Any,
        card_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect generation parameters"""
        params = {}
        
        # Check model card first
        if "generation" in card_data:
            if isinstance(card_data["generation"], dict):
                params.update(card_data["generation"])
                
        # Check config
        if config:
            if hasattr(config, "max_length"):
                params["max_length"] = config.max_length
            if hasattr(config, "do_sample"):
                params["do_sample"] = config.do_sample
            if hasattr(config, "temperature"):
                params["temperature"] = config.temperature
                
        # Set defaults if empty
        if not params:
            params = {
                "max_length": 128,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
            
        return params
        
    def _detect_special_tokens(self, config: Any) -> Dict[str, Any]:
        """Detect special tokens configuration"""
        tokens = {}
        
        if config:
            if hasattr(config, "bos_token_id"):
                tokens["bos_token_id"] = config.bos_token_id
            if hasattr(config, "eos_token_id"):
                tokens["eos_token_id"] = config.eos_token_id
            if hasattr(config, "pad_token_id"):
                tokens["pad_token_id"] = config.pad_token_id
            if hasattr(config, "sep_token_id"):
                tokens["sep_token_id"] = config.sep_token_id
                
        return tokens
        
    def _get_required_packages(
        self,
        arch_details: Dict[str, Any],
        task_type: str
    ) -> List[str]:
        """Get required packages based on architecture and task"""
        packages = set(arch_details.get("packages", ["transformers", "torch"]))
        
        # Add task-specific packages
        task_packages = {
            "text-to-image": ["diffusers", "invisible-watermark"],
            "image-to-text": ["pillow"],
            "automatic-speech-recognition": ["librosa"],
            "text-to-speech": ["soundfile"],
            "video-classification": ["decord"]
        }
        
        if task_type in task_packages:
            packages.update(task_packages[task_type])
            
        # Add common utilities
        packages.update(["accelerate", "safetensors"])
        
        return sorted(list(packages))