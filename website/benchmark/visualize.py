import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

class BenchmarkVisualizer:
    """Visualize benchmark results"""
    
    def __init__(self, results_dir: str = 'benchmark_results'):
        self.results_dir = Path(results_dir)
        
    def load_results(self, model_id: str) -> Dict:
        """Load benchmark results for a model"""
        file_path = self.results_dir / f"{model_id.replace('/', '_')}_benchmark.json"
        with open(file_path) as f:
            return json.load(f)
            
    def create_throughput_plot(self, results: Dict) -> go.Figure:
        """Create throughput vs batch size plot"""
        batch_sizes = []
        throughputs = []
        
        for batch_size, metrics in results['batches'].items():
            batch_sizes.append(int(batch_size))
            throughputs.append(metrics['throughput'])
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=batch_sizes,
            y=throughputs,
            mode='lines+markers',
            name='Throughput'
        ))
        
        fig.update_layout(
            title=f"Throughput vs Batch Size - {results['model_id']}",
            xaxis_title="Batch Size",
            yaxis_title="Throughput (images/sec)",
            template="plotly_white"
        )
        
        return fig
    
    def create_latency_plot(self, results: Dict) -> go.Figure:
        """Create latency vs batch size plot"""
        batch_sizes = []
        avg_latencies = []
        p95_latencies = []
        
        for batch_size, metrics in results['batches'].items():
            batch_sizes.append(int(batch_size))
            avg_latencies.append(metrics['avg_latency'] * 1000)  # Convert to ms
            p95_latencies.append(metrics['p95_latency'] * 1000)  # Convert to ms
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=batch_sizes,
            y=avg_latencies,
            mode='lines+markers',
            name='Average Latency'
        ))
        fig.add_trace(go.Scatter(
            x=batch_sizes,
            y=p95_latencies,
            mode='lines+markers',
            name='P95 Latency'
        ))
        
        fig.update_layout(
            title=f"Latency vs Batch Size - {results['model_id']}",
            xaxis_title="Batch Size",
            yaxis_title="Latency (ms)",
            template="plotly_white"
        )
        
        return fig
    
    def create_memory_plot(self, results: Dict) -> go.Figure:
        """Create memory usage vs batch size plot"""
        batch_sizes = []
        memory_usage = []
        gpu_memory_usage = []
        
        for batch_size, metrics in results['batches'].items():
            batch_sizes.append(int(batch_size))
            memory_usage.append(metrics['peak_memory_mb'])
            gpu_memory_usage.append(metrics['peak_gpu_memory_mb'])
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=batch_sizes,
            y=memory_usage,
            name='System Memory'
        ))
        
        if results['device'] == 'cuda':
            fig.add_trace(go.Bar(
                x=batch_sizes,
                y=gpu_memory_usage,
                name='GPU Memory'
            ))
            
        fig.update_layout(
            title=f"Memory Usage vs Batch Size - {results['model_id']}",
            xaxis_title="Batch Size",
            yaxis_title="Memory (MB)",
            template="plotly_white",
            barmode='group'
        )
        
        return fig
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, go.Figure]:
        """Compare performance across different models"""
        throughputs = {model_id: [] for model_id in model_ids}
        latencies = {model_id: [] for model_id in model_ids}
        batch_sizes = []
        
        # Load data for all models
        for model_id in model_ids:
            results = self.load_results(model_id)
            for batch_size, metrics in results['batches'].items():
                if int(batch_size) not in batch_sizes:
                    batch_sizes.append(int(batch_size))
                throughputs[model_id].append(metrics['throughput'])
                latencies[model_id].append(metrics['avg_latency'] * 1000)
        
        # Create comparison plots
        throughput_fig = go.Figure()
        latency_fig = go.Figure()
        
        for model_id in model_ids:
            throughput_fig.add_trace(go.Scatter(
                x=batch_sizes,
                y=throughputs[model_id],
                mode='lines+markers',
                name=model_id
            ))
            latency_fig.add_trace(go.Scatter(
                x=batch_sizes,
                y=latencies[model_id],
                mode='lines+markers',
                name=model_id
            ))
            
        throughput_fig.update_layout(
            title="Throughput Comparison",
            xaxis_title="Batch Size",
            yaxis_title="Throughput (images/sec)",
            template="plotly_white"
        )
        
        latency_fig.update_layout(
            title="Latency Comparison",
            xaxis_title="Batch Size",
            yaxis_title="Average Latency (ms)",
            template="plotly_white"
        )
        
        return {
            'throughput': throughput_fig,
            'latency': latency_fig
        }
    
    def generate_html_report(self, model_id: str, output_path: str = None):
        """Generate an HTML report with all plots"""
        results = self.load_results(model_id)
        
        # Create all plots
        throughput_plot = self.create_throughput_plot(results)
        latency_plot = self.create_latency_plot(results)
        memory_plot = self.create_memory_plot(results)
        
        # Combine into HTML
        html_content = f"""
        <html>
        <head>
            <title>Benchmark Report - {model_id}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Benchmark Report for {model_id}</h1>
            <div id="throughput"></div>
            <div id="latency"></div>
            <div id="memory"></div>
            <script>
                {throughput_plot.to_json()}
                Plotly.newPlot('throughput', throughput_plot.data, throughput_plot.layout);
                
                {latency_plot.to_json()}
                Plotly.newPlot('latency', latency_plot.data, latency_plot.layout);
                
                {memory_plot.to_json()}
                Plotly.newPlot('memory', memory_plot.data, memory_plot.layout);
            </script>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return html_content

def generate_benchmark_report(model_id: str, output_path: str = None):
    """Convenience function to generate a benchmark report"""
    visualizer = BenchmarkVisualizer()
    return visualizer.generate_html_report(model_id, output_path)

if __name__ == '__main__':
    # Example usage
    model_id = "microsoft/resnet-50"
    generate_benchmark_report(model_id, "benchmark_report.html")