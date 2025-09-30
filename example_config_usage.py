#!/usr/bin/env python3
"""
Example Usage of Configuration System

This script demonstrates how to use the JSON configuration system
for the 3-stage lung nodule detection pipeline.

Usage:
    python example_config_usage.py
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader, create_sample_config
from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline


def example_create_config():
    """Example of creating a configuration file"""
    print("="*60)
    print("EXAMPLE: Creating Configuration File")
    print("="*60)
    
    # Create sample configuration
    create_sample_config("my_config.json")
    print("✅ Sample configuration created: my_config.json")
    
    # Show how to edit it
    print("\nEdit the configuration file with your actual paths:")
    print("""
{
  "models": {
    "stage1": {
      "model_path": "/path/to/your/cancer_classifier.pth"
    },
    "stage2": {
      "model_path": "/path/to/your/lungrads_model.pth",
      "encoder_path": "/path/to/your/encoder.pkl",
      "scaler_path": "/path/to/your/scaler.pkl",
      "label_classes_path": "/path/to/your/label_classes.pkl"
    },
    "stage3": {
      "model_path": "/path/to/your/segmentation_model.pth"
    }
  },
  "data": {
    "input_ct_scan": "/path/to/your/ct_scan.nii.gz",
    "output_directory": "/path/to/results/"
  },
  "patient_info": {
    "age": 65.0,
    "gender": 1.0,
    "smoking_status": 0.0
  },
  "settings": {
    "device": "auto",
    "segmentation_threshold": 0.5,
    "verbose": true
  }
}
""")


def example_load_and_validate_config():
    """Example of loading and validating configuration"""
    print("\n" + "="*60)
    print("EXAMPLE: Loading and Validating Configuration")
    print("="*60)
    
    # Create a test configuration
    test_config = {
        "models": {
            "stage1": {
                "model_path": "models/stage1_cancer_classifier.pth"
            },
            "stage2": {
                "model_path": "models/stage2_lungrads_classifier.pth",
                "encoder_path": "models/encoder.pkl",
                "scaler_path": "models/scaler.pkl",
                "label_classes_path": "models/label_classes.pkl"
            },
            "stage3": {
                "model_path": "models/stage3_cancer_segmenter.pth"
            }
        },
        "data": {
            "input_ct_scan": "data/ct_scan.nii.gz",
            "output_directory": "results/"
        },
        "patient_info": {
            "age": 65.0,
            "gender": 1.0,
            "smoking_status": 0.0
        },
        "settings": {
            "device": "auto",
            "segmentation_threshold": 0.5,
            "verbose": True
        }
    }
    
    # Save test configuration
    with open("test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    print("✅ Test configuration created: test_config.json")
    
    try:
        # Load configuration
        config = ConfigLoader("test_config.json")
        
        # Print summary
        config.print_config_summary()
        
        # Validate paths
        config.print_validation_results()
        
        # Show how to access individual values
        print(f"\nAccessing individual values:")
        print(f"Stage 1 Model: {config.get_stage1_model_path()}")
        print(f"Stage 2 Encoder: {config.get_stage2_encoder_path()}")
        print(f"Patient Age: {config.get_patient_info()['age']}")
        print(f"Device: {config.get_device()}")
        print(f"Tabular Features: {config.get_tabular_features()}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")


def example_pipeline_with_config():
    """Example of using pipeline with configuration"""
    print("\n" + "="*60)
    print("EXAMPLE: Using Pipeline with Configuration")
    print("="*60)
    
    # Create a test configuration
    test_config = {
        "models": {
            "stage1": {
                "model_path": "models/stage1_cancer_classifier.pth"
            },
            "stage2": {
                "model_path": "models/stage2_lungrads_classifier.pth",
                "encoder_path": "models/encoder.pkl",
                "scaler_path": "models/scaler.pkl",
                "label_classes_path": "models/label_classes.pkl"
            },
            "stage3": {
                "model_path": "models/stage3_cancer_segmenter.pth"
            }
        },
        "data": {
            "input_ct_scan": "data/ct_scan.nii.gz",
            "output_directory": "results/"
        },
        "patient_info": {
            "age": 70.0,
            "gender": 0.0,
            "smoking_status": 1.0
        },
        "settings": {
            "device": "auto",
            "segmentation_threshold": 0.6,
            "verbose": True
        }
    }
    
    # Save test configuration
    with open("pipeline_test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    print("✅ Pipeline test configuration created: pipeline_test_config.json")
    
    try:
        # Load configuration
        config = ConfigLoader("pipeline_test_config.json")
        
        # Create pipeline from configuration
        print("\nCreating pipeline from configuration...")
        pipeline = LungNoduleDetectionPipeline.from_config(config)
        
        print("✅ Pipeline created successfully!")
        print(f"Device: {config.get_device()}")
        print(f"Segmentation Threshold: {config.get_segmentation_threshold()}")
        print(f"Patient Info: {config.get_patient_info()}")
        
        # Note: This would normally process a real CT scan
        print("\nNote: To actually process a CT scan, ensure all model files exist")
        print("and run: python run_with_config.py --config pipeline_test_config.json")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    if os.path.exists("pipeline_test_config.json"):
        os.remove("pipeline_test_config.json")


def example_command_line_usage():
    """Example of command line usage with configuration"""
    print("\n" + "="*60)
    print("EXAMPLE: Command Line Usage")
    print("="*60)
    
    print("1. Create configuration file:")
    print("   python example_config_usage.py  # This creates sample configs")
    print()
    
    print("2. Edit configuration with your paths:")
    print("   nano config.json  # or any text editor")
    print()
    
    print("3. Validate configuration:")
    print("   python run_with_config.py --config config.json --validate-only")
    print()
    
    print("4. Run pipeline:")
    print("   python run_with_config.py --config config.json")
    print()
    
    print("5. Alternative: Use main pipeline script:")
    print("   python lung_nodule_detection_pipeline.py --config config.json")
    print()
    
    print("6. Create sample configuration:")
    print("   python run_with_config.py --create-sample-config")


def example_python_usage():
    """Example of using configuration in Python code"""
    print("\n" + "="*60)
    print("EXAMPLE: Python Code Usage")
    print("="*60)
    
    print("""
# Load configuration
from config_loader import ConfigLoader
from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline

# Load configuration
config = ConfigLoader("config.json")

# Validate paths
config.print_validation_results()

# Create pipeline
pipeline = LungNoduleDetectionPipeline.from_config(config)

# Get patient info for custom processing
patient_info = config.get_patient_info()
tabular_features = config.get_tabular_features()

# Process CT scan
results = pipeline.process_ct_scan(
    config.get_input_ct_scan(),
    config.get_output_directory()
)

# Access results
print(f"Cancer detected: {results['stage1_results']['proceed_to_stage2']}")
if results['stage1_results']['proceed_to_stage2']:
    print(f"LungRADS: {results['stage2_results']['lungrads_label']}")
    print(f"Detections: {results['stage3_results']['num_detections']}")
""")


def main():
    """Main function with all examples"""
    print("3-STAGE LUNG NODULE DETECTION - CONFIGURATION SYSTEM EXAMPLES")
    
    # Show all examples
    example_create_config()
    example_load_and_validate_config()
    example_pipeline_with_config()
    example_command_line_usage()
    example_python_usage()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Create your configuration file with actual paths")
    print("2. Validate the configuration")
    print("3. Run the pipeline")
    print("\nFor more help:")
    print("  python run_with_config.py --help")
    print("  python lung_nodule_detection_pipeline.py --help")


if __name__ == "__main__":
    main()
