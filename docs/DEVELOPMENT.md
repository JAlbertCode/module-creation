# Development Guide

[Previous sections remain the same...]

## API Documentation

### Model Generation API

```python
POST /api/generate
{
    "modelUrl": "string",  # Hugging Face model URL
    "options": {           # Optional configuration
        "batchSize": "integer",
        "memory": "string",
        "gpu": "boolean",
        "optimizationLevel": "string"
    }
}

Response:
{
    "files": {
        "Dockerfile": "string",
        "requirements.txt": "string",
        "run_inference.py": "string",
        "module.yaml": "string",
        "README.md": "string"
    },
    "validation": {
        "status": "string",
        "errors": ["string"]
    }
}
```

### Monitoring API

```python
POST /api/monitor
{
    "modelId": "string",
    "days": "integer",     # Number of days of history
    "metrics": ["string"]  # Optional specific metrics to retrieve
}

Response:
{
    "statistics": {
        "throughput": {
            "mean": "float",
            "p95": "float",
            "trend": "string"
        },
        "latency": {
            "mean": "float",
            "p95": "float",
            "trend": "string"
        },
        "costs": {
            "mean": "float",
            "total": "float",
            "trend": "string"
        }
    },
    "plots": {
        "performance_over_time": {},
        "resource_usage": {},
        "cost_analysis": {}
    },
    "recommendations": ["string"]
}
```

### Benchmark API

```python
POST /api/benchmark
{
    "modelId": "string",
    "batchSizes": ["integer"],
    "iterations": "integer",
    "warmup": "integer"
}

Response:
{
    "results": {
        "throughput": ["float"],
        "latency": ["float"],
        "memory_usage": ["float"],
        "cost_per_inference": ["float"]
    },
    "analysis": {
        "optimal_batch_size": "integer",
        "resource_recommendations": ["string"]
    }
}
```

## WebSocket Events

### Performance Monitoring

```python
# Subscribe to real-time metrics
socket.emit('subscribe_metrics', {
    'modelId': 'string',
    'metrics': ['string']
})

# Receive metric updates
socket.on('metric_update', {
    'modelId': 'string',
    'timestamp': 'string',
    'metrics': {
        'throughput': 'float',
        'latency': 'float',
        'memory_usage': 'float'
    }
})

# Receive alerts
socket.on('alert', {
    'modelId': 'string',
    'type': 'string',
    'message': 'string',
    'severity': 'string'
})
```

## Environment Variables

```bash
# Server Configuration
FLASK_ENV=development|production
PORT=5000
HOST=0.0.0.0

# Database
DB_PATH=sqlite:///metrics.db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000

# Resource Limits
MAX_MEMORY=8Gi
MAX_CPU_CORES=4
ENABLE_GPU=true

# Monitoring
METRICS_RETENTION_DAYS=30
ALERT_WEBHOOK_URL=https://your-webhook-url
```

## Database Schema

### Metrics Table

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    model_id TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    throughput REAL NOT NULL,
    latency REAL NOT NULL,
    memory_mb REAL NOT NULL,
    gpu_memory_mb REAL,
    cost_per_inference REAL NOT NULL,
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX idx_metrics_model_time 
ON metrics(model_id, timestamp);
```

### Events Table

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    model_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX idx_events_model_time 
ON events(model_id, timestamp);
```

## Caching Strategy

### Redis Cache Structure

```python
# Model information cache
f"model_info:{model_id}" -> {
    "pipeline_type": "string",
    "description": "string",
    "tags": ["string"],
    "expire": 3600  # 1 hour
}

# Generated files cache
f"files:{model_id}" -> {
    "Dockerfile": "string",
    "requirements.txt": "string",
    "expire": 3600  # 1 hour
}

# Benchmark results cache
f"benchmark:{model_id}:{batch_size}" -> {
    "throughput": "float",
    "latency": "float",
    "expire": 1800  # 30 minutes
}
```

## Error Handling

### Error Codes

```python
ERROR_CODES = {
    'INVALID_MODEL': 1001,
    'GENERATION_FAILED': 1002,
    'VALIDATION_FAILED': 1003,
    'BENCHMARK_FAILED': 1004,
    'MONITORING_ERROR': 1005
}

# Error response format
{
    "error": {
        "code": "integer",
        "message": "string",
        "details": {}
    }
}
```

This development guide should provide a comprehensive reference for developers working on the project. Let me know if you'd like me to add any other sections or provide more details on specific aspects.