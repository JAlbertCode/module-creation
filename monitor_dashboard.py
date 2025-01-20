"""Real-time monitoring dashboard for deployed models"""

import dash
from dash import html, dcc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Any, List
import threading
import time
import psutil
import torch
from collections import deque

class MonitoringDashboard:
    """Real-time monitoring dashboard for model performance"""
    
    def __init__(self, db_path: str = 'monitoring.db'):
        self.db_path = db_path
        self.app = dash.Dash(__name__)
        self.metrics_buffer = {
            'timestamps': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'inference_time': deque(maxlen=100),
            'throughput': deque(maxlen=100)
        }
        self.setup_database()
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_database(self):
        """Initialize SQLite database for metrics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create metrics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_id TEXT,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_memory REAL,
                inference_time REAL,
                throughput REAL,
                errors INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.H1('Model Monitoring Dashboard',
                   style={'textAlign': 'center', 'padding': '20px'}),
            
            # Model Selection
            html.Div([
                html.Label('Select Model:'),
                dcc.Dropdown(
                    id='model-selector',
                    options=[],  # Will be populated dynamically
                    value=None
                )
            ], style={'padding': '20px'}),
            
            # Time Range Selection
            html.Div([
                html.Label('Time Range:'),
                dcc.RadioItems(
                    id='time-range',
                    options=[
                        {'label': 'Last Hour', 'value': '1H'},
                        {'label': 'Last Day', 'value': '1D'},
                        {'label': 'Last Week', 'value': '1W'}
                    ],
                    value='1H'
                )
            ], style={'padding': '20px'}),
            
            # Real-time Metrics
            html.Div([
                dcc.Graph(id='metrics-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,  # Update every 5 seconds
                    n_intervals=0
                )
            ]),
            
            # Resource Usage
            html.Div([
                dcc.Graph(id='resource-graph')
            ]),
            
            # Alerts and Recommendations
            html.Div([
                html.H3('Alerts and Recommendations'),
                html.Div(id='alerts-div')
            ], style={'padding': '20px'})
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks"""
        @self.app.callback(
            [dash.Output('metrics-graph', 'figure'),
             dash.Output('resource-graph', 'figure'),
             dash.Output('alerts-div', 'children')],
            [dash.Input('interval-component', 'n_intervals'),
             dash.Input('model-selector', 'value'),
             dash.Input('time-range', 'value')]
        )
        def update_graphs(n, model_id, time_range):
            if not model_id:
                return {}, {}, "No model selected"
                
            # Get time range
            end_time = datetime.now()
            if time_range == '1H':
                start_time = end_time - timedelta(hours=1)
            elif time_range == '1D':
                start_time = end_time - timedelta(days=1)
            else:  # 1W
                start_time = end_time - timedelta(weeks=1)
            
            # Get metrics from database
            metrics = self.get_metrics(model_id, start_time, end_time)
            
            # Create performance metrics graph
            metrics_fig = self.create_metrics_graph(metrics)
            
            # Create resource usage graph
            resource_fig = self.create_resource_graph(metrics)
            
            # Generate alerts and recommendations
            alerts = self.generate_alerts(metrics)
            
            return metrics_fig, resource_fig, alerts
    
    def get_metrics(self, model_id: str, 
                   start_time: datetime, 
                   end_time: datetime) -> pd.DataFrame:
        """Get metrics from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT *
            FROM metrics
            WHERE model_id = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=[model_id, start_time, end_time],
            parse_dates=['timestamp']
        )
        
        conn.close()
        return df
    
    def create_metrics_graph(self, metrics: pd.DataFrame) -> go.Figure:
        """Create performance metrics visualization"""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Inference Time', 'Throughput'))
        
        # Inference time
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['inference_time'],
                name='Inference Time',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Throughput
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['throughput'],
                name='Throughput',
                mode='lines'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        return fig
    
    def create_resource_graph(self, metrics: pd.DataFrame) -> go.Figure:
        """Create resource usage visualization"""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('CPU/Memory Usage', 'GPU Memory'))
        
        # CPU and Memory Usage
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['cpu_usage'],
                name='CPU Usage',
                mode='lines'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['memory_usage'],
                name='Memory Usage',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # GPU Memory if available
        if 'gpu_memory' in metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics['timestamp'],
                    y=metrics['gpu_memory'],
                    name='GPU Memory',
                    mode='lines'
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        return fig
    
    def generate_alerts(self, metrics: pd.DataFrame) -> html.Div:
        """Generate alerts and recommendations based on metrics"""
        alerts = []
        recommendations = []
        
        # Check for high resource usage
        recent = metrics.iloc[-10:]  # Last 10 readings
        
        if recent['cpu_usage'].mean() > 80:
            alerts.append("High CPU usage detected")
            recommendations.append("Consider scaling up CPU resources")
            
        if recent['memory_usage'].mean() > 80:
            alerts.append("High memory usage detected")
            recommendations.append("Consider increasing memory allocation")
            
        if 'gpu_memory' in metrics.columns and recent['gpu_memory'].mean() > 80:
            alerts.append("High GPU memory usage detected")
            recommendations.append("Consider GPU memory optimization")
            
        # Check for performance degradation
        if len(metrics) > 20:
            recent_perf = recent['inference_time'].mean()
            older_perf = metrics.iloc[-20:-10]['inference_time'].mean()
            
            if recent_perf > older_perf * 1.2:  # 20% degradation
                alerts.append("Performance degradation detected")
                recommendations.append("Check for system issues or conflicting processes")
        
        return html.Div([
            html.Div([
                html.H4('Alerts:', style={'color': 'red'}),
                html.Ul([html.Li(alert) for alert in alerts])
            ]) if alerts else html.Div(),
            
            html.Div([
                html.H4('Recommendations:', style={'color': 'blue'}),
                html.Ul([html.Li(rec) for rec in recommendations])
            ]) if recommendations else html.Div()
        ])
    
    def start_metric_collection(self, model_id: str):
        """Start collecting metrics for a model"""
        def collect_metrics():
            while True:
                try:
                    # Collect system metrics
                    cpu_usage = psutil.cpu_percent() / 100.0
                    memory_usage = psutil.virtual_memory().percent / 100.0
                    
                    # Collect GPU metrics if available
                    gpu_memory = None
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    
                    # Store metrics
                    conn = sqlite3.connect(self.db_path)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO metrics
                        (model_id, cpu_usage, memory_usage, gpu_memory)
                        VALUES (?, ?, ?, ?)
                    ''', (model_id, cpu_usage, memory_usage, gpu_memory))
                    conn.commit()
                    conn.close()
                    
                    time.sleep(5)  # Collect every 5 seconds
                    
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                    time.sleep(5)  # Wait before retrying
        
        # Start collection in background thread
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

def main():
    """Run the monitoring dashboard"""
    dashboard = MonitoringDashboard()
    dashboard.run_server()

if __name__ == '__main__':
    main()