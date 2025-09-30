#!/usr/bin/env python3
"""
Configuration Loader for 3-Stage Lung Nodule Detection Pipeline

This module loads configuration from JSON files and provides
easy access to all model paths, settings, and parameters.

Usage:
    from config_loader import ConfigLoader
    
    config = ConfigLoader("config.json")
    pipeline = LungNoduleDetectionPipeline.from_config(config)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader for the 3-stage pipeline"""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from: {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def get_stage1_model_path(self) -> str:
        """Get Stage 1 model path"""
        return self.config["models"]["stage1"]["model_path"]
    
    def get_stage2_model_path(self) -> Optional[str]:
        """Get Stage 2 model path"""
        return self.config["models"]["stage2"].get("model_path")
    
    def get_stage2_encoder_path(self) -> Optional[str]:
        """Get Stage 2 encoder path"""
        return self.config["models"]["stage2"].get("encoder_path")
    
    def get_stage2_scaler_path(self) -> Optional[str]:
        """Get Stage 2 scaler path"""
        return self.config["models"]["stage2"].get("scaler_path")
    
    def get_stage2_label_classes_path(self) -> Optional[str]:
        """Get Stage 2 label classes path"""
        return self.config["models"]["stage2"].get("label_classes_path")
    
    def get_stage2_encoders_and_scalers_path(self) -> Optional[str]:
        """Get Stage 2 combined encoders and scalers path"""
        return self.config["models"]["stage2"].get("encoders_and_scalers_path")
    
    def get_stage3_model_path(self) -> Optional[str]:
        """Get Stage 3 model path"""
        return self.config["models"]["stage3"].get("model_path")
    
    def get_input_ct_scan(self) -> str:
        """Get input CT scan path"""
        return self.config["data"]["input_ct_scan"]
    
    def get_output_directory(self) -> Optional[str]:
        """Get output directory path"""
        return self.config["data"].get("output_directory")
    
    def get_patient_info(self) -> Dict[str, float]:
        """Get patient information"""
        return self.config["patient_info"]
    
    def get_tabular_features(self) -> list:
        """Get tabular features as list"""
        patient_info = self.get_patient_info()
        return [patient_info["age"], patient_info["gender"], patient_info["smoking_status"]]
    
    def get_device(self) -> str:
        """Get device setting"""
        return self.config["settings"].get("device", "auto")
    
    def get_segmentation_threshold(self) -> float:
        """Get segmentation threshold"""
        return self.config["settings"].get("segmentation_threshold", 0.5)
    
    def get_verbose(self) -> bool:
        """Get verbose setting"""
        return self.config["settings"].get("verbose", False)
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all specified paths exist"""
        validation_results = {}
        
        # Stage 1
        stage1_path = self.get_stage1_model_path()
        validation_results["stage1_model"] = os.path.exists(stage1_path) if stage1_path else False
        
        # Stage 2
        stage2_path = self.get_stage2_model_path()
        validation_results["stage2_model"] = os.path.exists(stage2_path) if stage2_path else False
        
        encoder_path = self.get_stage2_encoder_path()
        validation_results["stage2_encoder"] = os.path.exists(encoder_path) if encoder_path else False
        
        scaler_path = self.get_stage2_scaler_path()
        validation_results["stage2_scaler"] = os.path.exists(scaler_path) if scaler_path else False
        
        label_classes_path = self.get_stage2_label_classes_path()
        validation_results["stage2_label_classes"] = os.path.exists(label_classes_path) if label_classes_path else False
        
        encoders_and_scalers_path = self.get_stage2_encoders_and_scalers_path()
        validation_results["stage2_encoders_and_scalers"] = os.path.exists(encoders_and_scalers_path) if encoders_and_scalers_path else False
        
        # Stage 3
        stage3_path = self.get_stage3_model_path()
        validation_results["stage3_model"] = os.path.exists(stage3_path) if stage3_path else False
        
        # Input data
        input_path = self.get_input_ct_scan()
        validation_results["input_ct_scan"] = os.path.exists(input_path) if input_path else False
        
        return validation_results
    
    def print_validation_results(self):
        """Print validation results for all paths"""
        print("\n" + "="*60)
        print("PATH VALIDATION RESULTS")
        print("="*60)
        
        validation_results = self.validate_paths()
        
        for path_name, exists in validation_results.items():
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"{path_name:25} : {status}")
        
        # Summary
        total_paths = len(validation_results)
        existing_paths = sum(validation_results.values())
        print(f"\nSummary: {existing_paths}/{total_paths} paths found")
        
        if existing_paths < total_paths:
            print("\n⚠️ Some paths are missing. Please check your configuration file.")
        else:
            print("\n✅ All specified paths exist!")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        return {
            "config_file": self.config_path,
            "stage1_model": self.get_stage1_model_path(),
            "stage2_model": self.get_stage2_model_path(),
            "stage2_encoder": self.get_stage2_encoder_path(),
            "stage2_scaler": self.get_stage2_scaler_path(),
            "stage2_label_classes": self.get_stage2_label_classes_path(),
            "stage2_encoders_and_scalers": self.get_stage2_encoders_and_scalers_path(),
            "stage3_model": self.get_stage3_model_path(),
            "input_ct_scan": self.get_input_ct_scan(),
            "output_directory": self.get_output_directory(),
            "patient_info": self.get_patient_info(),
            "device": self.get_device(),
            "segmentation_threshold": self.get_segmentation_threshold(),
            "verbose": self.get_verbose()
        }
    
    def print_config_summary(self):
        """Print a summary of the configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        summary = self.get_config_summary()
        
        print(f"Config File: {summary['config_file']}")
        print(f"Device: {summary['device']}")
        print(f"Segmentation Threshold: {summary['segmentation_threshold']}")
        print(f"Verbose: {summary['verbose']}")
        
        print(f"\nPatient Info:")
        patient_info = summary['patient_info']
        print(f"  Age: {patient_info['age']}")
        print(f"  Gender: {'Male' if patient_info['gender'] else 'Female'}")
        print(f"  Smoking: {'Yes' if patient_info['smoking_status'] else 'No'}")
        
        print(f"\nInput/Output:")
        print(f"  Input CT Scan: {summary['input_ct_scan']}")
        print(f"  Output Directory: {summary['output_directory']}")
        
        print(f"\nModel Paths:")
        print(f"  Stage 1 Model: {summary['stage1_model']}")
        print(f"  Stage 2 Model: {summary['stage2_model']}")
        print(f"  Stage 2 Encoder: {summary['stage2_encoder']}")
        print(f"  Stage 2 Scaler: {summary['stage2_scaler']}")
        print(f"  Stage 2 Label Classes: {summary['stage2_label_classes']}")
        print(f"  Stage 2 Encoders & Scalers: {summary['stage2_encoders_and_scalers']}")
        print(f"  Stage 3 Model: {summary['stage3_model']}")


def create_sample_config(output_path: str = "sample_config.json"):
    """Create a sample configuration file"""
    sample_config = {
        "models": {
            "stage1": {
                "model_path": "models/stage1_cancer_classifier.pth",
                "description": "Stage 1: Cancer Classification (ResNet50)"
            },
            "stage2": {
                "model_path": "models/stage2_lungrads_classifier.pth",
                "encoder_path": "models/encoder.pkl",
                "scaler_path": "models/scaler.pkl",
                "label_classes_path": "models/label_classes.pkl",
                "encoders_and_scalers_path": "models/encoders_and_scalers.pkl",
                "description": "Stage 2: LungRADS Classification"
            },
            "stage3": {
                "model_path": "models/stage3_cancer_segmenter.pth",
                "description": "Stage 3: Cancer Segmentation (FAUNet)"
            }
        },
        "data": {
            "input_ct_scan": "data/ct_scan.nii.gz",
            "output_directory": "results/",
            "description": "Input and output paths"
        },
        "patient_info": {
            "age": 65.0,
            "gender": 1.0,
            "smoking_status": 0.0,
            "description": "Patient information for LungRADS classification (age, gender: 1=male/0=female, smoking: 1=smoker/0=non-smoker)"
        },
        "settings": {
            "device": "auto",
            "segmentation_threshold": 0.5,
            "verbose": True,
            "description": "Pipeline settings"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration created: {output_path}")


def main():
    """Test the configuration loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Loader Test")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--create-sample", action="store_true", help="Create sample configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate paths in configuration")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    try:
        config = ConfigLoader(args.config)
        config.print_config_summary()
        
        if args.validate:
            config.print_validation_results()
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
