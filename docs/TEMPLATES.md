# Configuration Templates Guide

This guide explains how to use and customize configuration templates for different deployment scenarios.

## Available Templates

### 1. Production ResNet
Best for high-performance image classification in production environments.

```json
{
    "name": "production_resnet",
    "model_type": "image-classification",
    "resources": {
        "cpu": 4,
        "memory": "8Gi",
        "gpu": 1
    }
}
```

**When to use:**
- Production deployment
- High-throughput requirements
- GPU availability
- Large batch processing

### 2. Edge Deployment
Optimized for resource-constrained environments.

```json
{
    "name": "edge_deployment",
    "model_type": "image-classification",
    "resources": {
        "cpu": 1,
        "memory": "2Gi",
        "gpu": null
    }
}
```

**When to use:**
- Edge devices
- Limited resources
- Single-instance prediction
- CPU-only environments

### 3. BERT QA
Configured for question-answering models.

```json
{
    "name": "bert_qa",
    "model_type": "question-answering",
    "resources": {
        "cpu": 2,
        "memory": "4Gi"
    }
}
```

**When to use:**
- Question-answering tasks
- Medium-sized text processing
- Balanced resource usage

### 4. High Throughput
Maximum performance for text classification.

```json
{
    "name": "high_throughput",
    "model_type": "text-classification",
    "resources": {
        "cpu": 8,
        "memory": "16Gi",
        "gpu": 1
    }
}
```

**When to use:**
- High-volume text processing
- Production text classification
- Real-time requirements

## Customizing Templates

### Resource Configuration

1. CPU Allocation
```json
"resources": {
    "cpu": 4  // Number of CPU cores (1-32)
}
```

2. Memory Allocation
```json
"resources": {
    "memory": "8Gi"  // Memory in Gi (1Gi-64Gi)
}
```

3. GPU Configuration
```json
"resources": {
    "gpu": 1  // Number of GPUs (0-4)
}
```

### Model Settings

1. Batch Size
```json
"model": {
    "batch_size": 16  // Adjust based on memory and performance needs
}
```

2. Precision
```json
"model": {
    "half_precision": true  // Enable FP16 for better performance
}
```

3. Input Processing
```json
"input": {
    "preprocessing": {
        "resize": [224, 224],  // Image dimensions
        "normalize": true      // Enable normalization
    }
}
```

### Optimization Settings

1. TensorRT Optimization
```json
"optimization": {
    "enable_tensorrt": true,  // GPU optimization
    "dynamic_batching": true  // Dynamic batch handling
}
```

2. Caching
```json
"optimization": {
    "cache_size": "2Gi",     // Memory for caching
    "enable_cache": true     // Enable result caching
}
```

### Monitoring

1. Basic Monitoring
```json
"monitoring": {
    "enable_metrics": true,
    "log_level": "INFO"
}
```

2. Advanced Monitoring
```json
"monitoring": {
    "enable_metrics": true,
    "profiling": true,
    "log_level": "DEBUG",
    "export_traces": true
}
```

## Creating Custom Templates

1. Basic Template Structure
```json
{
    "name": "custom_template",
    "description": "Description of your template",
    "model_type": "your-model-type",
    "config": {
        "resources": {},
        "model": {},
        "input": {},
        "optimization": {},
        "monitoring": {}
    }
}
```

2. Save the template:
```bash
# Using the web interface
1. Click "Save Template"
2. Fill in template details
3. Click "Save"

# Using the API
POST /templates
Content-Type: application/json
{
    "name": "custom_template",
    "description": "Your description",
    "config": { ... }
}
```

## Best Practices

1. Resource Allocation
- Start with minimal resources
- Monitor usage patterns
- Scale up gradually
- Keep GPU optional

2. Batch Size Selection
- Start with batch_size=1
- Increase gradually
- Monitor memory usage
- Consider latency requirements

3. Optimization Tips
- Enable half_precision for GPU
- Use dynamic_batching for varying loads
- Enable caching for repeated queries
- Monitor and adjust cache size

4. Memory Management
- Account for model size
- Add buffer for preprocessing
- Consider peak usage scenarios
- Monitor swap usage

## Validation

Templates are automatically validated for:
1. Resource limits
2. Configuration compatibility
3. Security considerations
4. Best practices

Common validation errors:
```json
{
    "errors": [
        "CPU cores must be at least 1",
        "Memory must be specified in Gi",
        "Invalid model_type specified"
    ],
    "warnings": [
        "High memory allocation",
        "Large batch size may impact latency"
    ]
}
```

## Importing/Exporting

Export template:
```bash
GET /templates/export/{template_name}
```

Import template:
```bash
POST /templates/import
Content-Type: multipart/form-data
file: template.json
```

## Advanced Usage

1. Conditional Configuration
```json
{
    "config": {
        "resources": {
            "gpu": {"if_available": 1, "fallback": null},
            "cpu": {"min": 1, "preferred": 4}
        }
    }
}
```

2. Environment-Specific Settings
```json
{
    "config": {
        "environments": {
            "production": {
                "resources": {"cpu": 8, "memory": "16Gi"}
            },
            "development": {
                "resources": {"cpu": 2, "memory": "4Gi"}
            }
        }
    }
}
```

## Troubleshooting

Common issues and solutions:

1. Resource Allocation
   - Issue: Out of memory
   - Solution: Reduce batch size or increase memory

2. Performance
   - Issue: High latency
   - Solution: Enable optimizations or adjust batch size

3. GPU Usage
   - Issue: GPU not detected
   - Solution: Check CUDA installation and drivers

4. Template Validation
   - Issue: Invalid configuration
   - Solution: Check the validation errors and adjust accordingly