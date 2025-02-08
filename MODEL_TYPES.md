# Supported Model Types

This document lists all model types supported by the Hugging Face to Lilypad converter.

## Text Models

### Language Generation
- Causal Language Models (GPT, LLaMA, etc.)
- Text-to-Text Generation (T5, BART)
- Code Generation (CodeGen, StarCoder)
- Story Generation
- Dialog Systems

### Classification & Analysis
- Text Classification
- Sentiment Analysis
- Topic Classification
- Language Detection
- Hate Speech Detection
- Toxicity Detection

### Information Extraction
- Named Entity Recognition (NER)
- Part-of-Speech Tagging
- Dependency Parsing
- Event Extraction
- Keyword Extraction

### Question Answering
- Open Domain QA
- Reading Comprehension
- Closed Domain QA
- Multiple Choice QA

### Other Text Tasks
- Text Summarization
- Machine Translation
- Sentence Similarity
- Paraphrasing
- Grammar Correction
- Text Style Transfer

## Image Models

### Classification & Detection
- Image Classification
- Object Detection
  - YOLO Models
  - DETR Models
  - Faster R-CNN
- Face Detection
- Landmark Detection
- Person Detection

### Segmentation
- Semantic Segmentation
- Instance Segmentation
- Panoptic Segmentation
- Medical Image Segmentation

### Generation & Editing
- Text-to-Image Generation
  - Stable Diffusion
  - DALL-E Style Models
- Image-to-Image Translation
- Style Transfer
- Super Resolution
- Image Inpainting
- Image Outpainting
- Image Colorization

### Analysis & Understanding
- Depth Estimation
- Pose Estimation
- Face Recognition
- Facial Expression Analysis
- Scene Understanding
- Image Captioning

## Audio Models

### Speech Processing
- Speech Recognition (ASR)
- Text-to-Speech (TTS)
- Speaker Diarization
- Voice Conversion
- Speech Enhancement
- Accent Transfer

### Audio Analysis
- Audio Classification
- Sound Event Detection
- Music Classification
- Emotion Recognition from Speech
- Language Identification
- Speaker Verification

### Generation
- Music Generation
- Sound Generation
- Voice Cloning
- Audio Super Resolution
- Audio Source Separation

## Video Models

### Analysis
- Video Classification
- Action Recognition
- Activity Recognition
- Scene Understanding
- Motion Analysis
- Gesture Recognition

### Generation & Manipulation
- Text-to-Video Generation
- Video Frame Interpolation
- Video Super Resolution
- Video Colorization
- Style Transfer
- Video Inpainting

### Understanding
- Video Captioning
- Video Question Answering
- Video Summarization
- Video Object Tracking
- Video Object Segmentation
- Temporal Action Localization

## Multimodal Models

### Vision-Language
- Vision-Language Models (BLIP, LLaVA)
- Visual Question Answering
- Image-Text Matching
- Visual Reasoning
- Visual Dialog
- Visual Story Generation

### Document Understanding
- Document Layout Analysis (LayoutLM)
- Document Question Answering
- Document Information Extraction
- Table Understanding
- Form Understanding
- Receipt/Invoice Processing

### Audio-Visual
- Audio-Visual Speech Recognition
- Audio-Visual Event Localization
- Audio-Visual Navigation
- Cross-Modal Retrieval
- Sound Source Localization
- Lip Reading

## Specialized Models

### Scientific Computing
- Molecular Property Prediction
- Protein Structure Prediction
- Chemical Reaction Prediction
- Drug Discovery
- Material Science
- Climate Modeling

### Graph Models
- Graph Classification
- Node Classification
- Link Prediction
- Graph Generation
- Knowledge Graph Completion
- Graph Embedding

### 3D Processing
- Point Cloud Classification
- Point Cloud Segmentation
- 3D Object Detection
- 3D Shape Generation
- Neural Radiance Fields (NeRF)
- Mesh Generation

### Time Series
- Time Series Forecasting
- Anomaly Detection
- Event Prediction
- Trend Analysis
- Sequence Classification
- Time Series Generation

### Tabular Data
- Tabular Classification
- Tabular Regression
- Missing Value Imputation
- Outlier Detection
- Feature Selection
- Data Generation

## Local Development and Testing

Every model type supports both local testing and Lilypad deployment:

### Local Testing
```bash
# Build Docker image
docker build -t my-model .

# Run locally with mounted volumes
docker run -v /path/to/inputs:/inputs -v /path/to/outputs:/outputs my-model
```

### Lilypad Deployment
```bash
# Deploy to Lilypad network
lilypad run github.com/your-username/your-module:tag
```

## Implementation Status

- ✅ Fully Implemented
- 🟡 In Progress
- 📝 Planned

### Text Models
- ✅ Language Generation (Basic models)
- ✅ Text Classification
- 🟡 Sentiment Analysis
- 📝 Other Text Tasks

### Image Models
- ✅ Image Classification
- ✅ Text-to-Image Generation (SDXL)
- 🟡 Object Detection
- 📝 Segmentation
- 📝 Image Editing

### Vision-Language Models
- ✅ UI Understanding (UI-TARS)
- ✅ Vision-Language Instruction (Qwen-VL)
- 🟡 Visual Question Answering
- 📝 Document Understanding
- 📝 Image-Text Retrieval

### Video Models
- ✅ Text-to-Video Generation (SDV, HunyuanVideo)
- 📝 Video Analysis
- 📝 Video Understanding

### Audio Models
- 📝 All Audio Processing Tasks

### Multimodal Models
- 🟡 Vision-Language Models
- 📝 Document Understanding
- 📝 Audio-Visual Tasks

### Specialized Models
- 📝 Scientific Computing
- 🟡 Graph Models
- 📝 3D Processing
- 🟡 Time Series
- 📝 Tabular Data