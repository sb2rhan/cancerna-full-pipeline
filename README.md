# 3-Stage Lung Nodule Detection Pipeline

A comprehensive system for detecting and analyzing lung nodules in CT scans through three sequential stages: cancer classification, LungRADS classification, and cancer segmentation.

## Overview

This system processes 3D CT scans through three stages:

1. **Stage 1: Cancer Classification** - Binary classification (Cancer/No Cancer) using ResNet50
2. **Stage 2: LungRADS Classification** - Multi-class classification (1, 2, 3, 4A, 4B, 4X) using 3D CNN
3. **Stage 3: Cancer Segmentation** - Precise segmentation with coordinates and diameter extraction using FAUNet

## System Architecture

```
CT Scan Input
     ↓
Stage 1: Cancer Classification (ResNet50)
     ↓ (if cancer detected)
Stage 2: LungRADS Classification (3D CNN)
     ↓
Stage 3: Cancer Segmentation (FAUNet)
     ↓
Results: Coordinates, Diameter, Segmentation Mask
```

## Features

- **Modular Design**: Each stage can be used independently or as part of the full pipeline
- **Comprehensive Analysis**: From binary classification to precise segmentation
- **LungRADS Compliance**: Follows LungRADS guidelines for nodule assessment
- **Detailed Output**: Provides coordinates, diameters, and confidence scores
- **Visualization**: Built-in visualization tools for results
- **Batch Processing**: Support for processing multiple CT scans

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ct_scan_classifier_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA for GPU acceleration:
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

## Quick Start

### Full Pipeline Usage

```bash
python lung_nodule_detection_pipeline.py \
    --input /path/to/ct_scan.nii.gz \
    --stage1-model /path/to/stage1_model.pth \
    --stage2-model /path/to/stage2_model.pth \
    --stage3-model /path/to/stage3_model.pth \
    --output /path/to/results/
```

### Individual Stage Usage

```python
from lungrads_classifier import LungRADSClassifier
from cancer_segmentation import CancerSegmentation

# Stage 2: LungRADS Classification
classifier = LungRADSClassifier()
lungrads_class, lungrads_label, confidence = classifier.classify(
    ct_file_path, 
    tabular_features=[65.0, 1.0, 0.0]  # [age, gender, smoking_status]
)

# Stage 3: Cancer Segmentation
segmenter = CancerSegmentation()
results = segmenter.segment(ct_file_path, threshold=0.5)
```

## Detailed Usage

### Stage 1: Cancer Classification

Binary classification to determine if cancer is present in the CT scan.

**Input**: 3D CT scan (NIfTI, DICOM, or other medical imaging formats)
**Output**: Cancer probability, prediction, and decision to proceed to Stage 2

**Model**: ResNet50 3D (from existing classification_models.py)

### Stage 2: LungRADS Classification

Multi-class classification following LungRADS guidelines.

**Classes**:
- **1**: Benign - No follow-up needed
- **2**: Probably benign - 6-month follow-up recommended
- **3**: Probably malignant - 3-month follow-up recommended
- **4A**: Suspicious - 1-month follow-up or biopsy recommended
- **4B**: Suspicious - Biopsy recommended
- **4X**: Suspicious with additional features - Immediate evaluation recommended

**Input**: 3D CT scan + optional tabular features (age, gender, smoking status)
**Output**: LungRADS class, label, and confidence score

**Model**: Custom 3D CNN with tabular feature fusion

### Stage 3: Cancer Segmentation

Precise segmentation of lung cancer with detailed analysis.

**Input**: 3D CT scan
**Output**: 
- Number of detections
- Coordinates (pixel and world coordinates)
- Diameter measurements (in mm)
- Confidence scores
- Segmentation masks

**Model**: FAUNet (Fuzzy Attention U-Net) based on the notebook implementation

## File Structure

```
ct_scan_classifier_project/
├── lung_nodule_detection_pipeline.py    # Main pipeline orchestrator
├── lungrads_classifier.py               # Stage 2: LungRADS classification
├── cancer_segmentation.py               # Stage 3: Cancer segmentation
├── classification_models.py             # Stage 1: Cancer classification models
├── generate_transforms.py               # Data preprocessing transforms
├── unet3d.py                           # 3D U-Net implementation
├── utils.py                            # Utility functions
├── example_usage_3stage.py             # Usage examples
├── requirements.txt                     # Dependencies
├── README_3STAGE.md                    # This file
└── models/                             # Model weights (create this directory)
    ├── stage1_cancer_classifier.pth
    ├── stage2_lungrads_classifier.pth
    └── stage3_cancer_segmenter.pth
```

## Configuration

Create a `pipeline_config.json` file for custom configuration:

```json
{
  "pipeline": {
    "stage1_model_path": "models/stage1_cancer_classifier.pth",
    "stage2_model_path": "models/stage2_lungrads_classifier.pth",
    "stage3_model_path": "models/stage3_cancer_segmenter.pth",
    "device": "auto"
  },
  "preprocessing": {
    "ct_normalization": {
      "hu_min": -1000,
      "hu_max": 400
    },
    "resize": {
      "stage1": [64, 64, 64],
      "stage2": [64, 64, 64],
      "stage3": [512, 512]
    }
  },
  "segmentation": {
    "threshold": 0.5,
    "min_region_size": 20,
    "overlap_threshold": 0.3
  },
  "lungrads": {
    "tabular_features": {
      "age": 65.0,
      "gender": 1.0,
      "smoking_status": 0.0
    }
  }
}
```

## Examples

### Example 1: Basic Usage

```python
from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline

# Initialize pipeline
pipeline = LungNoduleDetectionPipeline(
    stage1_model_path="models/stage1_model.pth"
)

# Process CT scan
results = pipeline.process_ct_scan("path/to/ct_scan.nii.gz", "output/")

# Print results
print(f"Cancer detected: {results['stage1_results']['proceed_to_stage2']}")
if results['stage1_results']['proceed_to_stage2']:
    print(f"LungRADS: {results['stage2_results']['lungrads_label']}")
    print(f"Detections: {results['stage3_results']['num_detections']}")
```

### Example 2: Individual Stage Usage

```python
from lungrads_classifier import LungRADSClassifier

# Initialize LungRADS classifier
classifier = LungRADSClassifier("models/stage2_model.pth")

# Classify with patient information
tabular_features = [70.0, 0.0, 1.0]  # 70-year-old female smoker
lungrads_class, lungrads_label, confidence = classifier.classify(
    "path/to/ct_scan.nii.gz", 
    tabular_features
)

print(f"LungRADS: {lungrads_label} (confidence: {confidence:.3f})")
```

### Example 3: Batch Processing

```python
import os
from pathlib import Path

# Process multiple CT scans
ct_directory = "data/ct_scans/"
output_directory = "results/"

pipeline = LungNoduleDetectionPipeline("models/stage1_model.pth")

for ct_file in Path(ct_directory).glob("*.nii.gz"):
    results = pipeline.process_ct_scan(str(ct_file), output_directory)
    print(f"Processed {ct_file.name}: {results['overall_success']}")
```

## Output Format

The pipeline returns a comprehensive results dictionary:

```python
{
    "input_file": "path/to/ct_scan.nii.gz",
    "stage1_results": {
        "cancer_probability": 0.85,
        "prediction": "Cancer",
        "proceed_to_stage2": True
    },
    "stage2_results": {
        "lungrads_class": 3,
        "lungrads_label": "3",
        "confidence": 0.78
    },
    "stage3_results": {
        "num_detections": 2,
        "detections": [
            {
                "slice_index": 45,
                "center_pixel": (256, 128),
                "center_world": (12.5, -8.3, 45.2),
                "diameter_mm": 8.5,
                "area_pixels": 156,
                "confidence": 0.92,
                "bbox": (120, 240, 140, 260)
            }
        ],
        "success": True
    },
    "overall_success": True
}
```

## Model Training

### Stage 1: Cancer Classification
Use the existing training scripts in the project to train the ResNet50 model for binary cancer classification.

### Stage 2: LungRADS Classification
Train the LungRADS classifier using the `lungrads_classifier.py` module:

```python
# Example training setup
from lungrads_classifier import LungRADSClassifier

# The model architecture is defined in the class
# You would need to implement training loop with your dataset
```

### Stage 3: Cancer Segmentation
Train the FAUNet model using the implementation from the notebook:

```python
# The FAUNet architecture is defined in cancer_segmentation.py
# Training would follow the notebook implementation
```

## Performance Considerations

- **GPU Memory**: Stage 3 (segmentation) requires the most GPU memory
- **Processing Time**: Full pipeline typically takes 30-60 seconds per CT scan
- **Batch Processing**: Process multiple scans sequentially for memory efficiency
- **Model Loading**: Models are loaded once and reused for multiple scans

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check model file paths and formats
3. **Input Format Issues**: Ensure CT scans are in supported formats (NIfTI, DICOM)

### Debug Mode

Enable verbose output for debugging:

```bash
python lung_nodule_detection_pipeline.py \
    --input /path/to/ct_scan.nii.gz \
    --stage1-model /path/to/model.pth \
    --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{lung_nodule_detection_3stage,
  title={3-Stage Lung Nodule Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/lung-nodule-detection}
}
```

## Acknowledgments

- MONAI for medical imaging utilities
- PyTorch for deep learning framework
- SimpleITK for medical image processing
- The original FAUNet implementation from the notebook
