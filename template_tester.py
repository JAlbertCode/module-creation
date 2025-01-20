"""Template testing and validation utilities"""

import time
import resource
import json
import torch
import psutil
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class TestResult:
    success: bool
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class TemplateTester:
    """Test configuration templates with real workloads"""
    
    def __init__(self):
        self.memory_threshold = 0.9  # 90% memory usage warning
        self.cpu_threshold = 0.8     # 80% CPU usage warning
    
    def test_template(self, template: Dict[str, Any], model_id: str) -> TestResult:
        """Test a template configuration with a given model"""
        errors = []
        warnings = []
        recommendations = []
        metrics = {}
        
        try:
            # Test resource allocation
            resource_metrics = self._test_resources(template['config']['resources'])
            metrics.update(resource_metrics)
            
            # Test model loading and inference
            model_metrics = self._test_model(template['config'], model_id)
            metrics.update(model_metrics)
            
            # Analyze results
            self._analyze_results(metrics, warnings, recommendations)
            
            success = len(errors) == 0
            
        except Exception as e:
            success = False
            errors.append(str(e))
        
        return TestResult(success, metrics, errors, warnings, recommendations)
    
    def _test_resources(self, resources: Dict[str, Any]) -> Dict[str, float]:
        """Test resource allocation and usage"""
        metrics = {}
        
        # Test CPU allocation
        if 'cpu' in resources:
            cpu_count = resources['cpu']
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            metrics['cpu_usage'] = cpu_usage
            
            if cpu_usage > self.cpu_threshold:
                warnings.append(f"High CPU usage: {cpu_usage*100:.1f}%")
        
        # Test memory allocation
        if 'memory' in resources:
            memory = resources['memory']
            mem_gi = int(memory[:-2]) if memory.endswith('Gi') else 0
            mem_bytes = mem_gi * 1024 * 1024 * 1024
            
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            metrics['memory_usage'] = memory_usage / mem_bytes
            
            if metrics['memory_usage'] > self.memory_threshold:
                warnings.append(f"High memory usage: {metrics['memory_usage']*100:.1f}%")
        
        # Test GPU if requested
        if 'gpu' in resources and resources['gpu'] and torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                gpu_memory.append({
                    'device': i,
                    'allocated': memory_allocated,
                    'reserved': memory_reserved
                })
            metrics['gpu_memory'] = gpu_memory
        
        return metrics
    
    def _test_model(self, config: Dict[str, Any], model_id: str) -> Dict[str, float]:
        """Test model loading and inference"""
        metrics = {}
        
        # Load model
        start_time = time.time()
        pipe = pipeline(task=config.get('task', 'text-classification'), 
                       model=model_id,
                       device='cuda' if config.get('resources', {}).get('gpu') else 'cpu')
        metrics['model_load_time'] = time.time() - start_time
        
        # Test batch inference
        batch_size = config.get('model', {}).get('batch_size', 1)
        sample_input = self._get_sample_input(config, batch_size)
        
        # Warmup
        for _ in range(3):
            _ = pipe(sample_input)
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            _ = pipe(sample_input)
        metrics['average_inference_time'] = (time.time() - start_time) / 10
        
        return metrics
    
    def _get_sample_input(self, config: Dict[str, Any], batch_size: int) -> Any:
        """Generate sample input based on model type"""
        task = config.get('task', 'text-classification')
        
        if 'text' in task:
            return ['Sample text for testing.'] * batch_size
        elif 'image' in task:
            import numpy as np
            size = config.get('input', {}).get('preprocessing', {}).get('resize', [224, 224])
            return [np.random.randint(0, 255, (*size, 3), dtype=np.uint8)] * batch_size
        else:
            return [''] * batch_size
    
    def _analyze_results(self, metrics: Dict[str, float], 
                        warnings: List[str], 
                        recommendations: List[str]):
        """Analyze test results and generate recommendations"""
        
        # Analyze inference time
        if metrics.get('average_inference_time', 0) > 0.1:  # More than 100ms
            warnings.append("High inference latency detected")
            recommendations.append("Consider reducing batch size or increasing resources")
        
        # Analyze memory usage
        if metrics.get('memory_usage', 0) > 0.8:  # More than 80%
            warnings.append("High memory usage detected")
            recommendations.append("Consider increasing memory allocation")
        
        # Analyze GPU usage if available
        if 'gpu_memory' in metrics:
            for gpu in metrics['gpu_memory']:
                if gpu['allocated'] / gpu['reserved'] > 0.8:
                    warnings.append(f"High GPU memory usage on device {gpu['device']}")
                    recommendations.append("Consider using GPU memory optimizations")
    
    def generate_report(self, result: TestResult) -> str:
        """Generate a human-readable test report"""
        report = ["# Template Test Report\n"]
        
        # Status
        report.append("## Status")
        report.append(f"Status: {'Success' if result.success else 'Failed'}\n")
        
        # Metrics
        report.append("## Performance Metrics")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                report.append(f"- {metric}: {value:.3f}")
            else:
                report.append(f"- {metric}: {value}")
        report.append("")
        
        # Warnings
        if result.warnings:
            report.append("## Warnings")
            for warning in result.warnings:
                report.append(f"- {warning}")
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("## Recommendations")
            for rec in result.recommendations:
                report.append(f"- {rec}")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for error in result.errors:
                report.append(f"- {error}")
        
        return "\n".join(report)

def test_template(template_path: str, model_id: str) -> str:
    """Convenience function to test a template file"""
    with open(template_path) as f:
        template = json.load(f)
    
    tester = TemplateTester()
    result = tester.test_template(template, model_id)
    return tester.generate_report(result)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python template_tester.py template.json model_id")
        sys.exit(1)
    
    print(test_template(sys.argv[1], sys.argv[2]))