# API Reference

Complete API documentation for the Lilypad Model Generator.

## REST API

### Model Generation

#### Generate Module Files
```http
POST /api/generate
Content-Type: application/json
Authorization: Bearer <token>

{
    "modelUrl": "string",
    "options": {
        "batchSize": integer,
        "memory": "string",
        "gpu": boolean,
        "optimizationLevel": "string"
    }
}

Response: 200 OK
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

Error: 400 Bad Request
{
    "error": {
        "code": integer,
        "message": "string",
        "details": object
    }
}
```

#### Download Module Package
```http
POST /api/download
Content-Type: application/json
Authorization: Bearer <token>

{
    "modelUrl": "string"
}

Response: 200 OK
Content-Type: application/zip
[Binary ZIP file]

Error: 400 Bad Request
{
    "error": {
        "code": integer,
        "message": "string"
    }
}
```

### Monitoring

#### Get Model Metrics
```http
POST /api/monitor
Content-Type: application/json
Authorization: Bearer <token>

{
    "modelId": "string",
    "days": integer,
    "metrics": ["string"]
}

Response: 200 OK
{
    "statistics": {
        "throughput": {
            "mean": float,
            "p95": float,
            "trend": "string"
        },
        "latency": {
            "mean": float,
            "p95": float,
            "trend": "string"
        },
        "costs": {
            "mean": float,
            "total": float,
            "trend": "string"
        }
    },
    "plots": object,
    "recommendations": ["string"]
}
```

#### Record Metrics
```http
POST /api/monitor/record
Content-Type: application/json
Authorization: Bearer <token>

{
    "modelId": "string",
    "timestamp": "string",
    "metrics": {
        "throughput": float,
        "latency": float,
        "memory_usage": float,
        "gpu_memory_usage": float,
        "cost_per_inference": float
    }
}

Response: 200 OK
{
    "status": "success"
}
```

### Benchmarking

#### Run Benchmark
```http
POST /api/benchmark
Content-Type: application/json
Authorization: Bearer <token>

{
    "modelId": "string",
    "batchSizes": [integer],
    "iterations": integer,
    "warmup": integer
}

Response: 200 OK
{
    "results": {
        "throughput": [float],
        "latency": [float],
        "memory_usage": [float],
        "cost_per_inference": [float]
    },
    "analysis": {
        "optimal_batch_size": integer,
        "resource_recommendations": ["string"]
    }
}
```

## WebSocket API

### Connect
```javascript
const socket = io('ws://your-domain.com', {
    auth: {
        token: 'your-auth-token'
    }
});
```

### Subscribe to Metrics
```javascript
// Subscribe
socket.emit('subscribe_metrics', {
    modelId: 'string',
    metrics: ['string']
});

// Receive updates
socket.on('metric_update', (data) => {
    console.log(data);
    // {
    //     modelId: 'string',
    //     timestamp: 'string',
    //     metrics: {
    //         throughput: float,
    //         latency: float,
    //         memory_usage: float
    //     }
    // }
});
```

### Receive Alerts
```javascript
socket.on('alert', (data) => {
    console.log(data);
    // {
    //     modelId: 'string',
    //     type: 'string',
    //     message: 'string',
    //     severity: 'string'
    // }
});
```

## Client Libraries

### Python Client
```python
from lilypad_client import LilypadClient

client = LilypadClient(
    api_key='your-api-key',
    base_url='https://your-domain.com'
)

# Generate module
files = client.generate_module(
    model_url='https://huggingface.co/model-id',
    options={'batchSize': 4}
)

# Monitor metrics
metrics = client.get_metrics(
    model_id='model-id',
    days=30
)

# Run benchmark
results = client.run_benchmark(
    model_id='model-id',
    batch_sizes=[1, 2, 4, 8]
)
```

### JavaScript Client
```javascript
import { LilypadClient } from 'lilypad-client';

const client = new LilypadClient({
    apiKey: 'your-api-key',
    baseUrl: 'https://your-domain.com'
});

// Generate module
const files = await client.generateModule({
    modelUrl: 'https://huggingface.co/model-id',
    options: { batchSize: 4 }
});

// Monitor metrics
const metrics = await client.getMetrics({
    modelId: 'model-id',
    days: 30
});

// Run benchmark
const results = await client.runBenchmark({
    modelId: 'model-id',
    batchSizes: [1, 2, 4, 8]
});
```

## Error Codes

| Code | Description |
|------|-------------|
| 1001 | Invalid model URL |
| 1002 | Generation failed |
| 1003 | Validation failed |
| 1004 | Benchmark failed |
| 1005 | Monitoring error |
| 2001 | Authentication error |
| 2002 | Rate limit exceeded |
| 2003 | Permission denied |

## Rate Limits

- API requests: 100 per minute
- File generation: 10 per minute
- Benchmark runs: 5 per hour
- WebSocket connections: 10 per client

## Authentication

All requests must include an authentication token:
```http
Authorization: Bearer <your-token>
```

To obtain a token:
1. Register at https://your-domain.com/register
2. Generate an API key in your dashboard
3. Use the API key as your bearer token

## Versioning

API versions are specified in the URL:
```http
https://your-domain.com/v1/api/generate
```

Current versions:
- v1: Current stable version
- v2: Beta version (requires opt-in)

## Response Formats

All responses follow this format:
```json
{
    "data": {},      // Success response data
    "error": {},     // Error details (if any)
    "meta": {        // Metadata
        "version": "string",
        "timestamp": "string"
    }
}
```