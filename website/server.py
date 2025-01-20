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

# ... (previous server code) ...

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run benchmark for a model"""
    try:
        model_url = request.json.get('modelUrl')
        if not model_url:
            return jsonify({'error': 'No model URL provided'}), 400

        model_id = model_url.split('huggingface.co/')[-1]
        
        # Run benchmark
        benchmark_results = run_model_benchmark(model_id)
        
        # Generate report
        report_path = f'benchmark_results/{model_id.replace("/", "_")}_report.html'
        generate_benchmark_report(model_id, report_path)
        
        return jsonify({
            'results': benchmark_results,
            'reportUrl': f'/benchmark_reports/{os.path.basename(report_path)}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/benchmark_reports/<path:filename>')
def serve_report(filename):
    """Serve benchmark report files"""
    return send_file(f'benchmark_results/{filename}')