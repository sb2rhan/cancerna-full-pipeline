#!/usr/bin/env python3
"""
3-Stage Lung Nodule Detection Pipeline

This system processes 3D CT scans through three stages:
1. Cancer Classification (Binary: Cancer/No Cancer)
2. LungRADS Classification (1, 2, 3, 4A, 4B, 4X)
3. Lung Cancer Segmentation (Coordinates, Diameter, Segmentation Mask)

Usage:
    python lung_nodule_detection_pipeline.py --input /path/to/ct/scan.nii.gz --output /path/to/results/
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')
from classification_models import get_classification_model
from generate_transforms import generate_classification_test_transform
from lungrads_classifier import LungRADSClassifier
from cancer_segmentation import CancerSegmentation
from config_loader import ConfigLoader
import SimpleITK as sitk
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Stage1CancerClassifier:
    """Stage 1: Binary Cancer Classification using ResNet50"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model_path = model_path
        self.model = None
        self.transforms = None
        self._load_model()
        self._setup_transforms()
    
    def _load_model(self):
        """Load the trained ResNet50 classification model"""
        print("Loading Stage 1: Cancer Classification Model...")
        
        # Model configuration
        config = {
            "Model_name": "resnet50",
            "spatial_dims": 3,
            "n_input_channels": 1,
            "num_classes": 2
        }
        
        # Create model
        self.model = get_classification_model(
            Model_name=config["Model_name"],
            spatial_dims=config["spatial_dims"],
            n_input_channels=config["n_input_channels"],
            num_classes=config["num_classes"],
            device=self.device
        )
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Strip 'module.' prefix if it exists
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load state dict
            try:
                missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ Stage 1 model loaded successfully!")
                if missing_keys:
                    print(f"   Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"   Unexpected keys: {len(unexpected_keys)} keys")
            except Exception as e:
                print(f"⚠️ Error loading Stage 1 model: {e}")
        else:
            print("⚠️ Stage 1 model file not found, using random weights")
        
        self.model.eval()
    
    def _setup_transforms(self):
        """Setup data transforms for Stage 1"""
        self.transforms = generate_classification_test_transform(
            image_key="img",
            img_patch_size=[64, 64, 64]
        )
    
    def classify(self, ct_file_path: str) -> Tuple[float, str, bool, Optional[float], Optional[str]]:
        """
        Classify CT scan for cancer presence and malignancy when cancer present
        
        Args:
            ct_file_path: Path to CT scan file
            
        Returns:
            Tuple of (
                cancer_probability,
                cancer_prediction,            # "Cancer" or "No Cancer"
                proceed_to_stage2,
                malignancy_probability,       # if cancer present else None
                malignancy_prediction         # "Malignant"/"Benign" if cancer else None
            )
        """
        try:
            # Preprocess the file
            data = {"img": ct_file_path}
            processed_data = self.transforms(data)
            
            # Prepare input tensor
            input_tensor = processed_data["img"].unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(output)
                
                # Get probability of cancer class
                cancer_prob = probabilities[0, 1].cpu().item()
                
                # Determine prediction
                cancer_prediction = "Cancer" if cancer_prob > 0.5 else "No Cancer"
                proceed_to_stage2 = cancer_prob > 0.5

                # If cancer present, also provide malignancy assessment using same logits
                malignancy_probability: Optional[float] = None
                malignancy_prediction: Optional[str] = None
                if proceed_to_stage2:
                    # Interpret class-1 probability as malignant probability for Stage 1
                    malignancy_probability = cancer_prob
                    malignancy_prediction = "Malignant" if malignancy_probability > 0.5 else "Benign"
                
                return (
                    cancer_prob,
                    cancer_prediction,
                    proceed_to_stage2,
                    malignancy_probability,
                    malignancy_prediction,
                )
                
        except Exception as e:
            print(f"Error in Stage 1 classification: {e}")
            return 0.0, "Error", False, None, None


class Stage2LungRADSClassifier:
    """Stage 2: LungRADS Classification (1, 2, 3, 4A, 4B, 4X)"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 encoder_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 label_classes_path: Optional[str] = None,
                 encoders_and_scalers_path: Optional[str] = None,
                 device: str = 'auto'):
        self.classifier = LungRADSClassifier(
            model_path=model_path,
            encoder_path=encoder_path,
            scaler_path=scaler_path,
            label_classes_path=label_classes_path,
            encoders_and_scalers_path=encoders_and_scalers_path,
            device=device
        )
    
    def classify(self, ct_file_path: str, tabular_features: Optional[np.ndarray] = None) -> Tuple[int, str, float]:
        """
        Classify CT scan for LungRADS category
        
        Args:
            ct_file_path: Path to CT scan file
            tabular_features: Optional tabular features (age, gender, etc.)
            
        Returns:
            Tuple of (lungrads_class, lungrads_label, confidence)
        """
        return self.classifier.classify(ct_file_path, tabular_features)


class Stage3CancerSegmentation:
    """Stage 3: Lung Cancer Segmentation and Analysis"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.segmenter = CancerSegmentation(model_path=model_path, device=device)
    
    def segment(self, ct_file_path: str, threshold: float = 0.5) -> Dict:
        """
        Segment lung cancer and extract coordinates and diameter
        
        Args:
            ct_file_path: Path to CT scan file
            threshold: Segmentation threshold
            
        Returns:
            Dictionary containing segmentation results
        """
        return self.segmenter.segment(ct_file_path, threshold)

    def visualize_results(self, ct_file_path: str, results: Dict, save_path: Optional[str] = None):
        """Create and optionally save visualization of Stage 3 results."""
        return self.segmenter.visualize_results(ct_file_path, results, save_path)


class LungNoduleDetectionPipeline:
    """Main 3-Stage Lung Nodule Detection Pipeline"""
    
    def __init__(self, stage1_model_path: str, 
                 stage2_model_path: str = None,
                 stage2_encoder_path: str = None,
                 stage2_scaler_path: str = None,
                 stage2_label_classes_path: str = None,
                 stage2_encoders_and_scalers_path: str = None,
                 stage3_model_path: str = None, 
                 device: str = 'auto'):
        self.device = device
        
        # Initialize all stages
        self.stage1 = Stage1CancerClassifier(stage1_model_path, device)
        self.stage2 = Stage2LungRADSClassifier(
            model_path=stage2_model_path,
            encoder_path=stage2_encoder_path,
            scaler_path=stage2_scaler_path,
            label_classes_path=stage2_label_classes_path,
            encoders_and_scalers_path=stage2_encoders_and_scalers_path,
            device=device
        )
        self.stage3 = Stage3CancerSegmentation(stage3_model_path, device)
        
        print("✅ All 3 stages initialized successfully!")
    
    @classmethod
    def from_config(cls, config: ConfigLoader):
        """
        Create pipeline from configuration file
        
        Args:
            config: ConfigLoader instance
            
        Returns:
            LungNoduleDetectionPipeline instance
        """
        return cls(
            stage1_model_path=config.get_stage1_model_path(),
            stage2_model_path=config.get_stage2_model_path(),
            stage2_encoder_path=config.get_stage2_encoder_path(),
            stage2_scaler_path=config.get_stage2_scaler_path(),
            stage2_label_classes_path=config.get_stage2_label_classes_path(),
            stage2_encoders_and_scalers_path=config.get_stage2_encoders_and_scalers_path(),
            stage3_model_path=config.get_stage3_model_path(),
            device=config.get_device()
        )
    
    def process_ct_scan(self, ct_file_path: str, output_dir: str = None) -> Dict:
        """
        Process CT scan through all 3 stages
        
        Args:
            ct_file_path: Path to CT scan file
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing all results
        """
        results = {
            'input_file': ct_file_path,
            'stage1_results': {},
            'stage2_results': {},
            'stage3_results': {},
            'overall_success': False
        }
        
        print(f"\n{'='*60}")
        print(f"PROCESSING CT SCAN: {Path(ct_file_path).name}")
        print(f"{'='*60}")
        
        # Stage 1: Cancer Classification
        print("\n🔍 Stage 1: Cancer Classification...")
        cancer_prob, cancer_pred, proceed, mal_prob, mal_pred = self.stage1.classify(ct_file_path)
        results['stage1_results'] = {
            'cancer_probability': cancer_prob,
            'prediction': cancer_pred,
            'proceed_to_stage2': proceed,
            'malignancy_probability': mal_prob,
            'malignancy_prediction': mal_pred
        }
        
        print(f"   Cancer Probability: {cancer_prob:.4f}")
        print(f"   Prediction: {cancer_pred}")
        if proceed:
            print(f"   Malignancy: {mal_pred} ({(mal_prob if mal_prob is not None else 0.0):.4f})")
        print(f"   Proceed to Stage 2: {'Yes' if proceed else 'No'}")
        
        if not proceed:
            print("❌ No cancer detected. Stopping pipeline.")
            results['overall_success'] = True
            return results
        
        # Stage 2: LungRADS Classification
        print("\n🔍 Stage 2: LungRADS Classification...")
        lungrads_class, lungrads_label, lungrads_conf = self.stage2.classify(ct_file_path)
        results['stage2_results'] = {
            'lungrads_class': lungrads_class,
            'lungrads_label': lungrads_label,
            'confidence': lungrads_conf
        }
        
        print(f"   LungRADS Class: {lungrads_class}")
        print(f"   LungRADS Label: {lungrads_label}")
        print(f"   Confidence: {lungrads_conf:.4f}")
        
        # Stage 3: Cancer Segmentation
        print("\n🔍 Stage 3: Cancer Segmentation...")
        segmentation_results = self.stage3.segment(ct_file_path)
        results['stage3_results'] = segmentation_results
        
        print(f"   Number of Detections: {segmentation_results['num_detections']}")
        if segmentation_results['detections']:
            print("   Top Detections:")
            for i, detection in enumerate(segmentation_results['detections'][:3]):
                print(f"     {i+1}. Center: {detection['center_world']}")
                print(f"        Diameter: {detection['diameter_mm']:.2f} mm")
                print(f"        Confidence: {detection['confidence']:.4f}")
        
        results['overall_success'] = True
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(results, output_dir)
        
        print(f"\n✅ Pipeline completed successfully!")
        return results
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save results to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"results_{Path(results['input_file']).stem}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {json_path}")

        # Save Stage 3 labeled overlay image if detections exist
        try:
            stage3 = results.get('stage3_results', {})
            if stage3 and stage3.get('success') and stage3.get('detections'):
                overlay_path = os.path.join(
                    output_dir, f"overlay_{Path(results['input_file']).stem}.png"
                )
                self.stage3.visualize_results(results['input_file'], stage3, overlay_path)
                print(f"🖼️ Labeled cancer overlay saved to: {overlay_path}")
        except Exception as viz_err:
            print(f"⚠️ Could not save labeled overlay image: {viz_err}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="3-Stage Lung Nodule Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single CT scan
  python lung_nodule_detection_pipeline.py --input /path/to/scan.nii.gz --stage1-model /path/to/stage1_model.pth
  
  # Process with output directory
  python lung_nodule_detection_pipeline.py --input /path/to/scan.nii.gz --stage1-model /path/to/stage1_model.pth --output /path/to/results/
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Path to CT scan file"
    )
    
    parser.add_argument(
        "--stage1-model", "-m",
        help="Path to Stage 1 (cancer classification) model file"
    )
    
    parser.add_argument(
        "--stage2-model", "-m2",
        help="Path to Stage 2 (LungRADS classification) model file"
    )
    
    parser.add_argument(
        "--stage2-encoder", "-e2",
        help="Path to Stage 2 encoder pickle file"
    )
    
    parser.add_argument(
        "--stage2-scaler", "-s2",
        help="Path to Stage 2 scaler pickle file"
    )
    
    parser.add_argument(
        "--stage2-labels", "-l2",
        help="Path to Stage 2 label classes pickle file"
    )
    
    parser.add_argument(
        "--stage3-model", "-m3",
        help="Path to Stage 3 (cancer segmentation) model file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Directory to save results (optional)"
    )
    
    parser.add_argument(
        "--device", "-d",
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help="Device to use for inference (default: auto)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to JSON configuration file (overrides individual arguments)"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if using configuration file
        if args.config:
            print(f"Using configuration file: {args.config}")
            
            # Load configuration
            config = ConfigLoader(args.config)
            
            # Print configuration summary
            config.print_config_summary()
            
            # Validate paths
            config.print_validation_results()
            
            # Initialize pipeline from configuration
            pipeline = LungNoduleDetectionPipeline.from_config(config)
            
            # Get paths from configuration
            ct_file_path = config.get_input_ct_scan()
            output_dir = config.get_output_directory()
            
        else:
            # Use command line arguments
            print("Using command line arguments")
            
            # Validate required arguments
            if not args.input:
                print("Error: --input/-i argument is required when not using --config")
                sys.exit(1)
            
            if not args.stage1_model:
                print("Error: --stage1-model/-m argument is required when not using --config")
                sys.exit(1)
            
            # Validate inputs
            if not os.path.exists(args.input):
                print(f"Error: Input file does not exist: {args.input}")
                sys.exit(1)
            
            if not os.path.exists(args.stage1_model):
                print(f"Error: Stage 1 model file does not exist: {args.stage1_model}")
                sys.exit(1)
            
            # Initialize pipeline from arguments
            pipeline = LungNoduleDetectionPipeline(
                stage1_model_path=args.stage1_model,
                stage2_model_path=args.stage2_model,
                stage2_encoder_path=args.stage2_encoder,
                stage2_scaler_path=args.stage2_scaler,
                stage2_label_classes_path=args.stage2_labels,
                stage3_model_path=args.stage3_model,
                device=args.device
            )
            
            # Get paths from arguments
            ct_file_path = args.input
            output_dir = args.output
        
        # Process CT scan
        results = pipeline.process_ct_scan(ct_file_path, output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Input File: {Path(ct_file_path).name}")
        print(f"Stage 1 - Cancer: {results['stage1_results']['prediction']} ({results['stage1_results']['cancer_probability']:.4f})")
        if results['stage1_results'].get('proceed_to_stage2') and results['stage1_results'].get('malignancy_prediction') is not None:
            print(f"Stage 1 - Malignancy: {results['stage1_results']['malignancy_prediction']} ({(results['stage1_results']['malignancy_probability'] or 0.0):.4f})")
        
        if results['stage1_results']['proceed_to_stage2']:
            print(f"Stage 2 - LungRADS: {results['stage2_results']['lungrads_label']} ({results['stage2_results']['confidence']:.4f})")
            print(f"Stage 3 - Detections: {results['stage3_results']['num_detections']}")
        
        print(f"Overall Success: {'Yes' if results['overall_success'] else 'No'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
