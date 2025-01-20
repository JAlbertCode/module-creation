import time
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class PerformanceMonitor:
    """Monitor and track model performance over time"""
    
    def __init__(self, db_path: str = 'benchmark_metrics.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create metrics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DATETIME,
                model_id TEXT,
                batch_size INTEGER,
                throughput REAL,
                latency REAL,
                memory_mb REAL,
                gpu_memory_mb REAL,
                cost_per_inference REAL
            )
        ''')
        
        # Create events table for significant changes
        c.execute('''
            CREATE TABLE IF NOT EXISTS events (
                timestamp DATETIME,
                model_id TEXT,
                event_type TEXT,
                description TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction and magnitude"""
        if len(series) < 2:
            return "Insufficient data"
            
        # Calculate linear regression
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)
        
        # Calculate percentage change
        pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
        
        if abs(pct_change) < 1:
            return "Stable"
        elif pct_change > 0:
            return f"Improving ({pct_change:.1f}%)"
        else:
            return f"Degrading ({pct_change:.1f}%)"
    
    def generate_performance_report(self,
                                  model_id: str,
                                  days: int = 30) -> Dict:
        """Generate a comprehensive performance report"""
        metrics_df = self.get_metrics_history(model_id, days)
        events_df = self.get_events_history(model_id, days)
        
        if metrics_df.empty:
            return {
                "error": "No metrics data available for the specified period"
            }
        
        # Calculate statistics
        stats = {
            'throughput': {
                'mean': metrics_df['throughput'].mean(),
                'std': metrics_df['throughput'].std(),
                'trend': self._calculate_trend(metrics_df['throughput'])
            },
            'latency': {
                'mean': metrics_df['latency'].mean() * 1000,  # Convert to ms
                'p95': metrics_df['latency'].quantile(0.95) * 1000,
                'trend': self._calculate_trend(metrics_df['latency'])
            },
            'costs': {
                'mean': metrics_df['cost_per_inference'].mean(),
                'trend': self._calculate_trend(metrics_df['cost_per_inference']),
                'total_cost': metrics_df['cost_per_inference'].sum()
            },
            'resources': {
                'avg_memory': metrics_df['memory_mb'].mean(),
                'peak_memory': metrics_df['memory_mb'].max(),
                'avg_gpu_memory': metrics_df['gpu_memory_mb'].mean(),
                'peak_gpu_memory': metrics_df['gpu_memory_mb'].max()
            }
        }
        
        # Generate plots
        plots = {
            'performance_over_time': self._create_performance_plot(metrics_df),
            'resource_usage': self._create_resource_plot(metrics_df),
            'cost_analysis': self._create_cost_plot(metrics_df)
        }
        
        # Analyze events
        events = []
        if not events_df.empty:
            for _, event in events_df.iterrows():
                events.append({
                    'timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'type': event['event_type'],
                    'description': event['description']
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_df, stats)
        
        return {
            'model_id': model_id,
            'period': f"Last {days} days",
            'statistics': stats,
            'plots': plots,
            'events': events,
            'recommendations': recommendations
        }
    
    def _create_performance_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create performance visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Throughput', 'Latency'),
            vertical_spacing=0.15
        )
        
        # Throughput plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['throughput'],
                mode='lines+markers',
                name='Throughput'
            ),
            row=1, col=1
        )
        
        # Latency plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['latency'] * 1000,
                mode='lines+markers',
                name='Latency (ms)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Performance Metrics Over Time"
        )
        
        return fig
    
    def _create_resource_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create resource usage visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_mb'],
            mode='lines',
            name='Memory Usage (MB)'
        ))
        
        if df['gpu_memory_mb'].max() > 0:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['gpu_memory_mb'],
                mode='lines',
                name='GPU Memory Usage (MB)'
            ))
        
        fig.update_layout(
            title="Resource Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Memory (MB)",
            height=400
        )
        
        return fig
    
    def _create_cost_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create cost analysis visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cost_per_inference'],
            mode='lines',
            name='Cost per Inference ($)'
        ))
        
        # Add trend line
        z = np.polyfit(range(len(df)), df['cost_per_inference'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=p(range(len(df))),
            mode='lines',
            name='Cost Trend',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title="Cost Analysis Over Time",
            xaxis_title="Time",
            yaxis_title="Cost per Inference ($)",
            height=400
        )
        
        return fig
    
    def _generate_recommendations(self, 
                                df: pd.DataFrame, 
                                stats: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Throughput recommendations
        if stats['throughput']['trend'].startswith('Degrading'):
            recommendations.append(
                "Throughput is degrading. Consider investigating system bottlenecks "
                "or scaling resources."
            )
        
        # Latency recommendations
        if stats['latency']['p95'] > 100:  # If P95 latency > 100ms
            recommendations.append(
                "High P95 latency detected. Consider optimizing batch size or "
                "upgrading hardware."
            )
        
        # Cost recommendations
        if stats['costs']['trend'].startswith('Degrading'):
            recommendations.append(
                "Cost efficiency is decreasing. Review resource allocation and "
                "consider optimization techniques like quantization or pruning."
            )
        
        # Resource recommendations
        memory_utilization = stats['resources']['peak_memory'] / stats['resources']['avg_memory']
        if memory_utilization > 2:
            recommendations.append(
                "High memory usage variability detected. Consider implementing "
                "better memory management or batch size optimization."
            )
        
        return recommendations

if __name__ == '__main__':
    # Example usage
    monitor = PerformanceMonitor()
    report = monitor.generate_performance_report("microsoft/resnet-50")
    print(json.dumps(report, indent=2))