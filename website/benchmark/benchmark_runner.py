import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_id: str
    batch_size: int
    inference_times: List[float]
    memory_usage: List[float]
    gpu_memory_usage: List[float]
    throughput: float
    avg_latency: float
    p95_latency: float
    peak_memory: float
    peak_gpu_memory: float

class ModelBenchmark:
    """Benchmark different models for performance metrics"""
    
    def __init__(self, save_dir: str = 'benchmark_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run_benchmark(
        self,
        model_id: str,
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmark for a model with different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            # Initialize metrics
            inference_times = []
            memory_usage = []
            gpu_memory_usage = []
            
            # Load model
            model = pipeline(
                'image-classification',
                model=model_id,
                device=self.device
            )
            
            # Create dummy input (assuming image input)
            dummy_input = self._create_dummy_input(batch_size)
            
            # Warmup
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
            
            # Benchmark iterations
            for _ in range(num_iterations):
                # Record memory before inference
                memory_start = psutil.Process().memory_info().rss / 1024 / 1024
                if self.device == 'cuda':
                    gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    gpu_memory_start = 0
                
                # Run inference and time it
                start_time = time.perf_counter()
                _ = model(dummy_input)
                inference_time = time.perf_counter() - start_time
                
                # Record metrics
                inference_times.append(inference_time)
                memory_usage.append(memory_start)
                gpu_memory_usage.append(gpu_memory_start)
            
            # Calculate statistics
            avg_latency = np.mean(inference_times)
            p95_latency = np.percentile(inference_times, 95)
            throughput = batch_size / avg_latency
            peak_memory = max(memory_usage)
            peak_gpu_memory = max(gpu_memory_usage)
            
            # Store results
            results[batch_size] = BenchmarkResult(
                model_id=model_id,
                batch_size=batch_size,
                inference_times=inference_times,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory_usage,
                throughput=throughput,
                avg_latency=avg_latency,
                p95_latency=p95_latency,
                peak_memory=peak_memory,
                peak_gpu_memory=peak_gpu_memory
            )
            
        # Save results
        self._save_results(model_id, results)
        return results
    
    def _create_dummy_input(self, batch_size: int) -> List[np.ndarray]:
        """Create dummy input data for benchmarking"""
        # Create a dummy RGB image
        dummy_image = np.random.randint(
            0, 255, 
            (224, 224, 3), 
            dtype=np.uint8
        )
        return [dummy_image] * batch_size
    
    def _save_results(self, model_id: str, results: Dict[int, BenchmarkResult]):
        """Save benchmark results to JSON"""
        results_dict = {
            'model_id': model_id,
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'batches': {}
        }
        
        for batch_size, result in results.items():
            results_dict['batches'][str(batch_size)] = {
                'throughput': result.throughput,
                'avg_latency': result.avg_latency,
                'p95_latency': result.p95_latency,
                'peak_memory_mb': result.peak_memory,
                'peak_gpu_memory_mb': result.peak_gpu_memory
            }
        
        output_file = self.save_dir / f"{model_id.replace('/', '_')}_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
    def generate_report(self, results: Dict[int, BenchmarkResult]) -> str:
        """Generate a human-readable benchmark report"""
        report = [
            f"# Benchmark Report for {results[1].model_id}",
            f"\nDevice: {self.device}",
            "\n## Performance Metrics\n"
        ]
        
        # Create table header
        report.append("| Batch Size | Throughput (img/s) | Avg Latency (ms) | P95 Latency (ms) | Peak Memory (MB) | Peak GPU Memory (MB) |")
        report.append("|------------|-------------------|-----------------|-----------------|-----------------|-------------------|")
        
        # Add results for each batch size
        for batch_size, result in sorted(results.items()):
            report.append(
                f"| {batch_size} | {result.throughput:.2f} | {result.avg_latency*1000:.2f} | "
                f"{result.p95_latency*1000:.2f} | {result.peak_memory:.0f} | {result.peak_gpu_memory:.0f} |"
            )
        
        return '\n'.join(report)

def run_model_benchmark(model_id: str) -> str:
    """Run benchmark for a specific model and return the report"""
    benchmark = ModelBenchmark()
    results = benchmark.run_benchmark(model_id)
    return benchmark.generate_report(results)

if __name__ == '__main__':
    # Example usage
    model_id = "microsoft/resnet-50"
    print(run_model_benchmark(model_id))