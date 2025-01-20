# Quick Start Guide

This guide will help you quickly get started with the Lilypad Model Generator.

## 1. Deploy a Model in 5 Minutes

### Step 1: Open the Web Interface
```bash
# Start the server
docker run -p 5000:5000 lilypad-generator

# Visit in your browser
http://localhost:5000
```

### Step 2: Input Model URL
1. Copy your Hugging Face model URL (e.g., https://huggingface.co/microsoft/resnet-50)
2. Paste it into the web interface
3. Click "Generate Module Files"

### Step 3: Download and Deploy
1. Review the generated files
2. Download the zip package
3. Extract and deploy:
```bash
unzip lilypad-module.zip
cd lilypad-module
lilypad module deploy .
```

## 2. Monitor Performance

### View Dashboard
```bash
# Access monitoring dashboard
http://localhost:5000/monitor
```

### Key Metrics
- Throughput
- Latency
- Resource usage
- Costs
- Recommendations

## 3. Common Use Cases

### Image Classification
```bash
# Example deployment
docker run -e MODEL_ID=microsoft/resnet-50 \
          -e INPUT_PATH=/path/to/image.jpg \
          your-module-name
```

### Text Classification
```bash
# Example deployment
docker run -e MODEL_ID=bert-base-uncased \
          -e INPUT_PATH=/path/to/text.txt \
          your-module-name
```

### Object Detection
```bash
# Example deployment
docker run -e MODEL_ID=facebook/detr-resnet-50 \
          -e INPUT_PATH=/path/to/image.jpg \
          your-module-name
```

## 4. Troubleshooting

### Common Issues

1. Memory Errors
```bash
# Increase memory limit
docker run --memory=8g your-module-name
```

2. GPU Issues
```bash
# Enable GPU
docker run --gpus all your-module-name
```

3. Performance Issues
```bash
# Adjust batch size
docker run -e BATCH_SIZE=4 your-module-name
```

## 5. Best Practices

### Resource Optimization
- Start with small batch sizes
- Monitor memory usage
- Use GPU when available

### Cost Management
- Use monitoring dashboard
- Set up alerts
- Review recommendations

### Performance Tuning
- Benchmark different configurations
- Optimize batch sizes
- Monitor resource usage

## 6. Next Steps

1. Read the full documentation
2. Set up monitoring alerts
3. Explore advanced features
4. Join the community

## Need Help?

- Check the troubleshooting guide
- Review common issues
- Join our Discord community
- Open a GitHub issue

## Quick Reference

### Environment Variables
```bash
MODEL_ID=huggingface-model-id
INPUT_PATH=/path/to/input
BATCH_SIZE=1
GPU_ENABLED=true
MEMORY_LIMIT=8g
```

### Basic Commands
```bash
# Build image
docker build -t your-module-name .

# Run locally
docker run your-module-name

# Deploy to Lilypad
lilypad module deploy .

# Monitor metrics
lilypad module status your-module-name
```

### Useful Links
- [Full Documentation](./DEVELOPMENT.md)
- [API Reference](./API.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [GitHub Issues](https://github.com/your-repo/issues)