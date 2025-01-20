"""Visualization tools for template analysis"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from template_tester import TemplateTester
from template_compare import TemplateComparator

class TemplateVisualizer:
    """Generate visualizations for template analysis"""
    
    def __init__(self):
        self.tester = TemplateTester()
        self.comparator = TemplateComparator()
        
    def create_performance_dashboard(self, template: Dict[str, Any], 
                                   model_id: str) -> go.Figure:
        """Create an interactive performance dashboard"""
        
        # Run tests
        result = self.tester.test_template(template, model_id)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Resource Usage',
                'Performance Metrics',
                'Memory Profile',
                'Batch Processing'
            )
        )
        
        # Resource Usage gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=result.metrics.get('cpu_usage', 0) * 100,
                title={'text': "CPU Usage %"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Performance bar chart
        perf_metrics = {
            'Load Time': result.metrics.get('model_load_time', 0),
            'Inference Time': result.metrics.get('average_inference_time', 0)
        }
        fig.add_trace(
            go.Bar(
                x=list(perf_metrics.keys()),
                y=list(perf_metrics.values()),
                name="Time (seconds)"
            ),
            row=1, col=2
        )
        
        # Memory profile line
        if 'memory_profile' in result.metrics:
            profile = result.metrics['memory_profile']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(profile))),
                    y=profile,
                    name="Memory Usage"
                ),
                row=2, col=1
            )
        
        # Batch processing scatter
        if 'batch_times' in result.metrics:
            batch_sizes = result.metrics.get('batch_sizes', [])
            batch_times = result.metrics['batch_times']
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=batch_times,
                    mode='lines+markers',
                    name="Batch Processing"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Template Performance Dashboard"
        )
        
        return fig
    
    def create_comparison_dashboard(self, templates: List[Dict[str, Any]], 
                                  model_id: str) -> go.Figure:
        """Create a comparison dashboard for multiple templates"""
        
        # Run tests for all templates
        results = [self.tester.test_template(t, model_id) for t in templates]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Resource Usage Comparison',
                'Performance Comparison',
                'Memory Usage Comparison',
                'Cost Analysis'
            )
        )
        
        # Resource usage comparison
        cpu_usage = [r.metrics.get('cpu_usage', 0) * 100 for r in results]
        memory_usage = [r.metrics.get('memory_usage', 0) * 100 for r in results]
        
        fig.add_trace(
            go.Bar(
                name='CPU Usage %',
                x=[f'Template {i+1}' for i in range(len(templates))],
                y=cpu_usage
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Memory Usage %',
                x=[f'Template {i+1}' for i in range(len(templates))],
                y=memory_usage
            ),
            row=1, col=1
        )
        
        # Performance comparison
        inference_times = [r.metrics.get('average_inference_time', 0) for r in results]
        
        fig.add_trace(
            go.Bar(
                name='Inference Time',
                x=[f'Template {i+1}' for i in range(len(templates))],
                y=inference_times
            ),
            row=1, col=2
        )
        
        # Memory profile comparison
        for i, result in enumerate(results):
            if 'memory_profile' in result.metrics:
                fig.add_trace(
                    go.Scatter(
                        name=f'Template {i+1}',
                        y=result.metrics['memory_profile']
                    ),
                    row=2, col=1
                )
        
        # Cost analysis
        if all('cost_estimate' in r.metrics for r in results):
            costs = [r.metrics['cost_estimate'] for r in results]
            fig.add_trace(
                go.Bar(
                    name='Estimated Cost',
                    x=[f'Template {i+1}' for i in range(len(templates))],
                    y=costs
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Template Comparison Dashboard"
        )
        
        return fig
    
    def create_optimization_plot(self, template: Dict[str, Any], 
                               model_id: str) -> go.Figure:
        """Create visualization for optimization opportunities"""
        
        # Test template with different configurations
        base_result = self.tester.test_template(template, model_id)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        batch_results = []
        
        for batch_size in batch_sizes:
            test_template = template.copy()
            test_template['config']['model']['batch_size'] = batch_size
            result = self.tester.test_template(test_template, model_id)
            batch_results.append(result)
        
        # Create optimization plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Batch Size vs Throughput',
                'Memory vs Performance',
                'GPU Utilization',
                'Optimization Recommendations'
            )
        )
        
        # Batch size vs throughput
        throughputs = [1/r.metrics['average_inference_time'] for r in batch_results]
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=throughputs,
                mode='lines+markers',
                name='Throughput'
            ),
            row=1, col=1
        )
        
        # Memory vs performance
        memory_usage = [r.metrics.get('memory_usage', 0) for r in batch_results]
        fig.add_trace(
            go.Scatter(
                x=memory_usage,
                y=throughputs,
                mode='markers',
                name='Memory vs Throughput'
            ),
            row=1, col=2
        )
        
        # GPU utilization if available
        if 'gpu_memory' in base_result.metrics:
            gpu_util = [r.metrics['gpu_memory'][0]['allocated'] / 
                       r.metrics['gpu_memory'][0]['reserved'] 
                       for r in batch_results]
            fig.add_trace(
                go.Scatter(
                    x=batch_sizes,
                    y=gpu_util,
                    mode='lines+markers',
                    name='GPU Utilization'
                ),
                row=2, col=1
            )
        
        # Add recommendations
        optimal_batch = batch_sizes[np.argmax(throughputs)]
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=optimal_batch,
                delta={'reference': template['config']['model']['batch_size']},
                title="Recommended Batch Size",
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Optimization Analysis"
        )
        
        return fig
    
    def save_dashboard(self, fig: go.Figure, output_path: str):
        """Save dashboard as HTML file"""
        fig.write_html(output_path)
    
def create_visualization(template_path: str, model_id: str, output_path: str):
    """Convenience function to create and save visualization"""
    import json
    
    with open(template_path) as f:
        template = json.load(f)
    
    visualizer = TemplateVisualizer()
    dashboard = visualizer.create_performance_dashboard(template, model_id)
    visualizer.save_dashboard(dashboard, output_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python template_visualizer.py template.json model_id output.html")
        sys.exit(1)
    
    create_visualization(sys.argv[1], sys.argv[2], sys.argv[3])