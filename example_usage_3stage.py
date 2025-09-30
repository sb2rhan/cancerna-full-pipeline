#!/usr/bin/env python3
"""
Example Usage of 3-Stage Lung Nodule Detection Pipeline

This script demonstrates how to use the 3-stage lung nodule detection system
for processing CT scans through cancer classification, LungRADS classification,
and cancer segmentation.

Usage:
    python example_usage_3stage.py --input /path/to/ct/scan.nii.gz --stage1-model /path/to/stage1_model.pth
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline
from lungrads_classifier import LungRADSClassifier
from cancer_segmentation import CancerSegmentation


def example_single_stage_usage():
    """Example of using individual stages separately"""
    print("="*60)
    print("EXAMPLE: Using Individual Stages")
    print("="*60)
    
    # Example CT scan path (replace with actual path)
    ct_file_path = "path/to/your/ct_scan.nii.gz"
    
    # Stage 2: LungRADS Classification
    print("\n🔍 Stage 2: LungRADS Classification")
    print("-" * 40)
    
    lungrads_classifier = LungRADSClassifier()
    
    # Example tabular features: [age, gender, smoking_status]
    tabular_features = np.array([65.0, 1.0, 0.0])  # 65-year-old male non-smoker
    
    lungrads_class, lungrads_label, confidence = lungrads_classifier.classify(
        ct_file_path, tabular_features
    )
    
    print(f"LungRADS Class: {lungrads_class}")
    print(f"LungRADS Label: {lungrads_label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Get detailed explanation
    explanation = lungrads_classifier.explain_classification(ct_file_path, tabular_features)
    print(f"Interpretation: {explanation['interpretation']}")
    
    # Stage 3: Cancer Segmentation
    print("\n🔍 Stage 3: Cancer Segmentation")
    print("-" * 40)
    
    segmenter = CancerSegmentation()
    results = segmenter.segment(ct_file_path, threshold=0.5)
    
    print(f"Number of detections: {results['num_detections']}")
    print(f"Success: {results['success']}")
    
    if results['detections']:
        print("Top detections:")
        for i, detection in enumerate(results['detections'][:3]):
            print(f"  {i+1}. Slice {detection['slice_index']}: "
                  f"Center={detection['center_world']}, "
                  f"Diameter={detection['diameter_mm']:.2f}mm, "
                  f"Confidence={detection['confidence']:.4f}")


def example_full_pipeline_usage():
    """Example of using the full 3-stage pipeline"""
    print("="*60)
    print("EXAMPLE: Full 3-Stage Pipeline")
    print("="*60)
    
    # Example paths (replace with actual paths)
    ct_file_path = "path/to/your/ct_scan.nii.gz"
    stage1_model_path = "path/to/stage1_model.pth"
    stage2_model_path = "path/to/stage2_model.pth"  # Optional
    stage3_model_path = "path/to/stage3_model.pth"  # Optional
    output_dir = "results/"
    
    # Initialize pipeline
    pipeline = LungNoduleDetectionPipeline(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        stage3_model_path=stage3_model_path,
        device='auto'
    )
    
    # Process CT scan
    results = pipeline.process_ct_scan(ct_file_path, output_dir)
    
    # Print detailed results
    print_detailed_results(results)


def example_batch_processing():
    """Example of processing multiple CT scans"""
    print("="*60)
    print("EXAMPLE: Batch Processing")
    print("="*60)
    
    # Example directory containing CT scans
    ct_directory = "path/to/ct_scans/"
    stage1_model_path = "path/to/stage1_model.pth"
    output_dir = "batch_results/"
    
    # Initialize pipeline
    pipeline = LungNoduleDetectionPipeline(
        stage1_model_path=stage1_model_path,
        device='auto'
    )
    
    # Find all CT scan files
    ct_extensions = ['.nii.gz', '.nii', '.nrrd', '.dcm']
    ct_files = []
    
    for ext in ct_extensions:
        ct_files.extend(Path(ct_directory).glob(f"*{ext}"))
        ct_files.extend(Path(ct_directory).glob(f"**/*{ext}"))
    
    print(f"Found {len(ct_files)} CT scan files")
    
    # Process each file
    batch_results = []
    for i, ct_file in enumerate(ct_files):
        print(f"\nProcessing {i+1}/{len(ct_files)}: {ct_file.name}")
        
        try:
            results = pipeline.process_ct_scan(str(ct_file), output_dir)
            batch_results.append({
                'file': str(ct_file),
                'success': results['overall_success'],
                'cancer_detected': results['stage1_results']['proceed_to_stage2'],
                'lungrads_label': results['stage2_results'].get('lungrads_label', 'N/A'),
                'num_detections': results['stage3_results'].get('num_detections', 0)
            })
        except Exception as e:
            print(f"Error processing {ct_file.name}: {e}")
            batch_results.append({
                'file': str(ct_file),
                'success': False,
                'error': str(e)
            })
    
    # Save batch results
    batch_output_file = os.path.join(output_dir, "batch_results.json")
    with open(batch_output_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\nBatch processing completed. Results saved to: {batch_output_file}")
    
    # Print summary
    successful = sum(1 for r in batch_results if r['success'])
    cancer_detected = sum(1 for r in batch_results if r.get('cancer_detected', False))
    
    print(f"\nSummary:")
    print(f"  Total files: {len(batch_results)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Cancer detected: {cancer_detected}")


def print_detailed_results(results):
    """Print detailed results from pipeline processing"""
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    # Stage 1 Results
    print(f"\n🔍 Stage 1: Cancer Classification")
    print(f"  Cancer Probability: {results['stage1_results']['cancer_probability']:.4f}")
    print(f"  Prediction: {results['stage1_results']['prediction']}")
    print(f"  Proceed to Stage 2: {'Yes' if results['stage1_results']['proceed_to_stage2'] else 'No'}")
    
    if results['stage1_results']['proceed_to_stage2']:
        # Stage 2 Results
        print(f"\n🔍 Stage 2: LungRADS Classification")
        print(f"  LungRADS Class: {results['stage2_results']['lungrads_class']}")
        print(f"  LungRADS Label: {results['stage2_results']['lungrads_label']}")
        print(f"  Confidence: {results['stage2_results']['confidence']:.4f}")
        
        # Stage 3 Results
        print(f"\n🔍 Stage 3: Cancer Segmentation")
        print(f"  Number of Detections: {results['stage3_results']['num_detections']}")
        print(f"  Success: {results['stage3_results']['success']}")
        
        if results['stage3_results']['detections']:
            print(f"  Top Detections:")
            for i, detection in enumerate(results['stage3_results']['detections'][:5]):
                print(f"    {i+1}. Slice {detection['slice_index']}:")
                print(f"       Center (World): {detection['center_world']}")
                print(f"       Center (Pixel): {detection['center_pixel']}")
                print(f"       Diameter: {detection['diameter_mm']:.2f} mm")
                print(f"       Area: {detection['area_pixels']} pixels")
                print(f"       Confidence: {detection['confidence']:.4f}")
    
    print(f"\n✅ Overall Success: {'Yes' if results['overall_success'] else 'No'}")


def create_sample_config():
    """Create a sample configuration file"""
    config = {
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
                "gender": 1.0,  # 1=male, 0=female
                "smoking_status": 0.0  # 1=smoker, 0=non-smoker
            }
        }
    }
    
    config_file = "pipeline_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sample configuration created: {config_file}")


def main():
    """Main function with examples"""
    parser = argparse.ArgumentParser(
        description="3-Stage Lung Nodule Detection - Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--example",
        choices=['single_stage', 'full_pipeline', 'batch_processing', 'config'],
        default='full_pipeline',
        help="Example to run"
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Path to CT scan file (for single_stage and full_pipeline examples)"
    )
    
    parser.add_argument(
        "--stage1-model", "-m1",
        help="Path to Stage 1 model file"
    )
    
    parser.add_argument(
        "--stage2-model", "-m2",
        help="Path to Stage 2 model file"
    )
    
    parser.add_argument(
        "--stage3-model", "-m3",
        help="Path to Stage 3 model file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if args.example == 'single_stage':
        example_single_stage_usage()
    elif args.example == 'full_pipeline':
        if not args.input or not args.stage1_model:
            print("Error: --input and --stage1-model are required for full_pipeline example")
            sys.exit(1)
        example_full_pipeline_usage()
    elif args.example == 'batch_processing':
        example_batch_processing()
    elif args.example == 'config':
        create_sample_config()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nTo run the actual pipeline:")
    print("python lung_nodule_detection_pipeline.py --input /path/to/ct.nii.gz --stage1-model /path/to/model.pth")


if __name__ == "__main__":
    main()
