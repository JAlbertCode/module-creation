from flask import Flask, request, jsonify, send_file, render_template
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import tempfile
import docker
from model_templates import ModelTemplate
from config_validator import ConfigValidator
from benchmark.benchmark_runner import run_model_benchmark
from benchmark.visualize import generate_benchmark_report
from benchmark.monitor import PerformanceMonitor

app = Flask(__name__)

# Initialize monitoring
monitor = PerformanceMonitor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor_dashboard():
    return render_template('monitor.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of monitored models"""
    try:
        # Get models from the database
        conn = sqlite3.connect(monitor.db_path)
        c = conn.cursor()
        c.execute('SELECT DISTINCT model_id FROM metrics')
        models = [{'id': row[0]} for row in c.fetchall()]
        conn.close()
        
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor', methods=['POST'])
def get_metrics():
    """Get monitoring metrics for a model"""
    try:
        data = request.json
        model_id = data.get('modelId')
        days = int(data.get('days', 30))
        
        if not model_id:
            return jsonify({'error': 'Model ID required'}), 400
            
        report = monitor.generate_performance_report(model_id, days)
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/record', methods=['POST'])
def record_metrics():
    """Record new metrics"""
    try:
        data = request.json
        benchmark_results = data.get('results')
        
        if not benchmark_results:
            return jsonify({'error': 'Benchmark results required'}), 400
            
        monitor.record_metrics(benchmark_results)
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ... (previous endpoints remain the same) ...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)