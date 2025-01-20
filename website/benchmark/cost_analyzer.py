from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import yaml
from pathlib import Path

@dataclass
class CostEstimate:
    """Container for cost estimation results"""
    hourly_cost: float
    monthly_cost: float
    yearly_cost: float
    breakdown: Dict[str, float]
    recommendations: List[str]

class CostAnalyzer:
    """Analyze costs and provide optimization recommendations"""
    
    def __init__(self):
        self.cloud_prices = {
            'cpu_hour': 0.05,  # Price per CPU core hour
            'memory_gb_hour': 0.01,  # Price per GB of memory per hour
            'gpu_hour': 0.50,  # Price per GPU hour
            'storage_gb_month': 0.10,  # Price per GB of storage per month
            'network_gb': 0.10  # Price per GB of network transfer
        }
    
    def estimate_costs(self, 
                      benchmark_results: Dict,
                      monthly_requests: int = 1000000) -> CostEstimate:
        """Estimate operational costs based on benchmark results"""
        
        # Extract resource requirements
        avg_latency = float(benchmark_results['batches']['1']['avg_latency'])
        peak_memory_gb = float(benchmark_results['batches']['1']['peak_memory_mb']) / 1024
        gpu_required = benchmark_results['device'] == 'cuda'
        
        # Calculate resource utilization
        requests_per_second = 1 / avg_latency
        hours_per_month = (monthly_requests / requests_per_second) / 3600
        
        # Calculate costs
        cpu_cost = hours_per_month * self.cloud_prices['cpu_hour']
        memory_cost = hours_per_month * peak_memory_gb * self.cloud_prices['memory_gb_hour']
        gpu_cost = hours_per_month * self.cloud_prices['gpu_hour'] if gpu_required else 0
        
        # Storage and network costs
        model_size_gb = 1  # Approximate model size
        storage_cost = model_size_gb * self.cloud_prices['storage_gb_month']
        network_cost = (monthly_requests * 0.001) * self.cloud_prices['network_gb']  # Assuming 1MB per request
        
        # Total costs
        total_hourly = (cpu_cost + memory_cost + gpu_cost + storage_cost + network_cost) / 730  # hours per month
        total_monthly = cpu_cost + memory_cost + gpu_cost + storage_cost + network_cost
        total_yearly = total_monthly * 12
        
        # Cost breakdown
        breakdown = {
            'CPU': cpu_cost,
            'Memory': memory_cost,
            'GPU': gpu_cost,
            'Storage': storage_cost,
            'Network': network_cost
        }
        
        # Generate optimization recommendations
        recommendations = self._generate_recommendations(
            benchmark_results,
            breakdown,
            monthly_requests
        )
        
        return CostEstimate(
            hourly_cost=total_hourly,
            monthly_cost=total_monthly,
            yearly_cost=total_yearly,
            breakdown=breakdown,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                benchmark_results: Dict,
                                cost_breakdown: Dict,
                                monthly_requests: int) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Analyze batch size efficiency
        batch_metrics = benchmark_results['batches']
        max_throughput = max(float(m['throughput']) for m in batch_metrics.values())
        optimal_batch = max(batch_metrics.items(), 
                          key=lambda x: float(x[1]['throughput']))[0]
        
        recommendations.append(
            f"Optimal batch size for throughput: {optimal_batch} "
            f"(achieving {max_throughput:.2f} requests/second)"
        )
        
        # Memory optimization
        memory_gb = float(batch_metrics['1']['peak_memory_mb']) / 1024
        if memory_gb > 4:
            recommendations.append(
                "Consider using a smaller model or quantization to reduce "
                "memory footprint and costs"
            )
        
        # GPU recommendations
        if benchmark_results['device'] == 'cuda':
            if monthly_requests < 100000:
                recommendations.append(
                    "Given your current request volume, CPU deployment might be "
                    "more cost-effective than GPU"
                )
        
        # Batch size recommendations for cost optimization
        if monthly_requests > 1000000:
            recommendations.append(
                "For high request volumes, consider using larger batch sizes "
                "to improve cost efficiency"
            )
        
        # Autoscaling recommendations
        recommendations.append(
            "Implement autoscaling based on request patterns to optimize "
            "resource utilization"
        )
        
        return recommendations
    
    def generate_resource_config(self,
                               benchmark_results: Dict,
                               monthly_requests: int) -> Dict:
        """Generate recommended resource configuration"""
        
        batch_metrics = benchmark_results['batches']
        optimal_batch = max(batch_metrics.items(), 
                          key=lambda x: float(x[1]['throughput']))[0]
        
        peak_memory_gb = float(batch_metrics[optimal_batch]['peak_memory_mb']) / 1024
        requests_per_second = float(batch_metrics[optimal_batch]['throughput'])
        
        # Calculate required instances
        target_rps = monthly_requests / (30 * 24 * 3600)  # Average requests per second
        min_instances = max(1, int(target_rps / requests_per_second))
        max_instances = min_instances * 3  # Allow for traffic spikes
        
        config = {
            'resources': {
                'cpu': 2,
                'memory': f"{int(peak_memory_gb * 1.5)}Gi",  # Add 50% buffer
                'gpu': 1 if benchmark_results['device'] == 'cuda' else None
            },
            'autoscaling': {
                'minReplicas': min_instances,
                'maxReplicas': max_instances,
                'targetCPUUtilization': 70,
                'targetMemoryUtilization': 80
            },
            'batch': {
                'size': int(optimal_batch),
                'timeout': "100ms"
            }
        }
        
        return config

def generate_cost_report(benchmark_results: Dict, monthly_requests: int) -> str:
    """Generate a cost analysis report"""
    analyzer = CostAnalyzer()
    costs = analyzer.estimate_costs(benchmark_results, monthly_requests)
    config = analyzer.generate_resource_config(benchmark_results, monthly_requests)
    
    report = [
        "# Cost Analysis Report\n",
        "## Estimated Costs",
        f"- Hourly: ${costs.hourly_cost:.2f}",
        f"- Monthly: ${costs.monthly_cost:.2f}",
        f"- Yearly: ${costs.yearly_cost:.2f}\n",
        "## Cost Breakdown",
    ]
    
    for resource, cost in costs.breakdown.items():
        report.append(f"- {resource}: ${cost:.2f}/month")
    
    report.extend([
        "\n## Optimization Recommendations"
    ])
    
    for i, rec in enumerate(costs.recommendations, 1):
        report.append(f"{i}. {rec}")
    
    report.extend([
        "\n## Recommended Resource Configuration",
        "```yaml",
        yaml.dump(config, default_flow_style=False),
        "```"
    ])
    
    return '\n'.join(report)

if __name__ == '__main__':
    # Example usage
    with open('benchmark_results/microsoft_resnet-50_benchmark.json') as f:
        results = json.load(f)
    
    print(generate_cost_report(results, monthly_requests=1000000))