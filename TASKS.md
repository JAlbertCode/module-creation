# Task Tracking

## Current Coverage

✅ Currently Supported:
- Text Generation (CausalLM models)
- Diffusion Models
- Basic default fallback

## Required Model Type Support

### Text Models
✅ Text Classification (AutoModelForSequenceClassification)
✅ Token Classification (AutoModelForTokenClassification)
✅ Question Answering (AutoModelForQuestionAnswering)
✅ Summarization (AutoModelForSeq2SeqLM)
✅ Translation (AutoModelForSeq2SeqLM)
✅ Named Entity Recognition (AutoModelForTokenClassification)
✅ Text-to-Text Generation (AutoModelForSeq2SeqLM)
- [ ] Masked Language Models (AutoModelForMaskedLM)

### Vision Models
- [ ] Image Classification (AutoModelForImageClassification)
- [ ] Object Detection (AutoModelForObjectDetection)
- [ ] Image Segmentation (AutoModelForImageSegmentation)
- [ ] Depth Estimation (AutoModelForDepthEstimation)
- [ ] Image-to-Text (AutoModelForVision2Seq)
- [ ] Vision Transformers (ViTModel)

### Audio Models
- [ ] Speech Recognition (AutoModelForSpeechSeq2Seq)
- [ ] Audio Classification (AutoModelForAudioClassification)
- [ ] Speech-to-Text (Wav2Vec2, Whisper)
- [ ] Text-to-Speech (Bark, FastSpeech)
- [ ] Audio-to-Audio (AudioMAE)

### Video Models
- [ ] Video Classification (AutoModelForVideoClassification)
- [ ] Video-to-Text (AutoModelForVideoToText)
- [ ] Zero-shot Video Classification
- [ ] Action Recognition

### Multimodal Models
- [ ] Vision-Language Models (CLIPModel)
- [ ] Visual Question Answering (AutoModelForVisionQuestionAnswering)
- [ ] Document Question Answering (LayoutLMForQuestionAnswering)
- [ ] Image-Text Matching (AutoModelForImageTextRetrieval)

## Implementation Plan

1. Add Model Type Support:
   - Add each AutoModel class mapping
   - Define input/output types
   - Specify required processors
   - Add necessary dependencies
   - Update architecture detection

2. Test Each Addition:
   - Create test cases using real models
   - Verify input/output handling
   - Check resource management
   - Validate error handling

3. Document Coverage:
   - Update supported models list
   - Add usage examples
   - Document any limitations
   - Update progress tracking

## Progress Tracking

[2025-02-09] Text Models
✓ Added architecture mappings for BERT, RoBERTa, T5, BART
✓ Added input/output type specifications
✓ Added processor mappings
✓ Updated architecture detection

Format for tracking implementation:
```
[Date] Model Type
✓ Added architecture detection
✓ Added processor mapping
✓ Added test cases
✓ Updated documentation
```

## Next Steps

1. Implement Text Models support
2. Add test cases
3. Document new capabilities
4. Move to Vision Models