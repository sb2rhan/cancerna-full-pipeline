#!/usr/bin/env python3
"""
Example Usage with Pickle Files for LungRADS Classifier

This script demonstrates how to use the 3-stage pipeline with
encoder, scaler, and label classes pickle files for Stage 2.

Usage:
    python example_with_pickle_files.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline
from lungrads_classifier import LungRADSClassifier


def example_with_pickle_files():
    """Example using all pickle files for LungRADS classifier"""
    print("="*60)
    print("EXAMPLE: Using Pickle Files for LungRADS Classifier")
    print("="*60)
    
    # Example file paths (replace with your actual paths)
    ct_file_path = "path/to/your/ct_scan.nii.gz"
    
    # Stage 2 files
    stage2_model_path = "path/to/your/lungrads_model.pth"
    encoder_path = "path/to/your/encoder.pkl"
    scaler_path = "path/to/your/scaler.pkl"
    label_classes_path = "path/to/your/label_classes.pkl"
    
    print("\n🔍 Stage 2: LungRADS Classification with Pickle Files")
    print("-" * 50)
    
    # Initialize LungRADS classifier with all components
    classifier = LungRADSClassifier(
        model_path=stage2_model_path,
        encoder_path=encoder_path,
        scaler_path=scaler_path,
        label_classes_path=label_classes_path,
        device='auto'
    )
    
    # Example tabular features: [age, gender, smoking_status]
    tabular_features = np.array([65.0, 1.0, 0.0])  # 65-year-old male non-smoker
    
    print(f"Input tabular features: {tabular_features}")
    print("Processing with encoder and scaler...")
    
    # Classify with preprocessing
    lungrads_class, lungrads_label, confidence = classifier.classify(
        ct_file_path, tabular_features
    )
    
    print(f"\nResults:")
    print(f"LungRADS Class: {lungrads_class}")
    print(f"LungRADS Label: {lungrads_label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Get detailed explanation
    explanation = classifier.explain_classification(ct_file_path, tabular_features)
    print(f"\nInterpretation: {explanation['interpretation']}")
    
    # Show all class probabilities
    print(f"\nAll class probabilities:")
    for label, prob in explanation['all_probabilities'].items():
        print(f"  {label}: {prob:.4f}")


def example_full_pipeline_with_pickle_files():
    """Example using full pipeline with pickle files"""
    print("\n" + "="*60)
    print("EXAMPLE: Full Pipeline with Pickle Files")
    print("="*60)
    
    # Example file paths
    ct_file_path = "path/to/your/ct_scan.nii.gz"
    
    # Stage 1
    stage1_model_path = "path/to/your/cancer_classifier.pth"
    
    # Stage 2 with pickle files
    stage2_model_path = "path/to/your/lungrads_model.pth"
    encoder_path = "path/to/your/encoder.pkl"
    scaler_path = "path/to/your/scaler.pkl"
    label_classes_path = "path/to/your/label_classes.pkl"
    
    # Stage 3
    stage3_model_path = "path/to/your/segmentation_model.pth"
    
    # Initialize full pipeline
    pipeline = LungNoduleDetectionPipeline(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        stage2_encoder_path=encoder_path,
        stage2_scaler_path=scaler_path,
        stage2_label_classes_path=label_classes_path,
        stage3_model_path=stage3_model_path,
        device='auto'
    )
    
    # Process CT scan
    results = pipeline.process_ct_scan(ct_file_path, "results/")
    
    # Print results
    print_detailed_results(results)


def example_command_line_usage():
    """Example command line usage with pickle files"""
    print("\n" + "="*60)
    print("EXAMPLE: Command Line Usage with Pickle Files")
    print("="*60)
    
    print("Command line usage:")
    print()
    print("python lung_nodule_detection_pipeline.py \\")
    print("    --input /path/to/ct_scan.nii.gz \\")
    print("    --stage1-model /path/to/cancer_classifier.pth \\")
    print("    --stage2-model /path/to/lungrads_model.pth \\")
    print("    --stage2-encoder /path/to/encoder.pkl \\")
    print("    --stage2-scaler /path/to/scaler.pkl \\")
    print("    --stage2-labels /path/to/label_classes.pkl \\")
    print("    --stage3-model /path/to/segmentation_model.pth \\")
    print("    --output /path/to/results/")


def example_pickle_file_requirements():
    """Show what each pickle file should contain"""
    print("\n" + "="*60)
    print("PICKLE FILE REQUIREMENTS")
    print("="*60)
    
    print("\n1. Encoder pickle file (encoder.pkl):")
    print("   - Should contain a fitted encoder (e.g., LabelEncoder, OneHotEncoder)")
    print("   - Used to preprocess categorical features")
    print("   - Example: sklearn.preprocessing.LabelEncoder")
    
    print("\n2. Scaler pickle file (scaler.pkl):")
    print("   - Should contain a fitted scaler (e.g., StandardScaler, MinMaxScaler)")
    print("   - Used to normalize numerical features")
    print("   - Example: sklearn.preprocessing.StandardScaler")
    
    print("\n3. Label classes pickle file (label_classes.pkl):")
    print("   - Should contain the mapping of class indices to LungRADS labels")
    print("   - Can be a dictionary: {0: '1', 1: '2', 2: '3', 3: '4A', 4: '4B', 5: '4X'}")
    print("   - Or a list: ['1', '2', '3', '4A', '4B', '4X']")
    
    print("\nExample of creating these files:")
    print("""
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create encoder
encoder = LabelEncoder()
encoder.fit(['male', 'female'])  # Example categorical data
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Create scaler
scaler = StandardScaler()
scaler.fit([[65, 1, 0], [70, 0, 1]])  # Example numerical data
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create label classes
label_classes = {0: '1', 1: '2', 2: '3', 3: '4A', 4: '4B', 5: '4X'}
with open('label_classes.pkl', 'wb') as f:
    pickle.dump(label_classes, f)
""")


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
            for i, detection in enumerate(results['stage3_results']['detections'][:3]):
                print(f"    {i+1}. Slice {detection['slice_index']}:")
                print(f"       Center (World): {detection['center_world']}")
                print(f"       Diameter: {detection['diameter_mm']:.2f} mm")
                print(f"       Confidence: {detection['confidence']:.4f}")
    
    print(f"\n✅ Overall Success: {'Yes' if results['overall_success'] else 'No'}")


def main():
    """Main function with examples"""
    print("3-STAGE LUNG NODULE DETECTION - PICKLE FILES EXAMPLES")
    
    # Show pickle file requirements
    example_pickle_file_requirements()
    
    # Show command line usage
    example_command_line_usage()
    
    # Show individual usage
    example_with_pickle_files()
    
    # Show full pipeline usage
    example_full_pipeline_with_pickle_files()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nTo use with your pickle files:")
    print("1. Place your pickle files in the appropriate locations")
    print("2. Update the file paths in the examples above")
    print("3. Run the pipeline with the --stage2-encoder, --stage2-scaler, --stage2-labels arguments")


if __name__ == "__main__":
    main()
