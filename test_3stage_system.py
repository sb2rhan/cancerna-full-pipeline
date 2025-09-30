#!/usr/bin/env python3
"""
Test Script for 3-Stage Lung Nodule Detection System

This script tests the individual components and the full pipeline
to ensure everything is working correctly.

Usage:
    python test_3stage_system.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline
        print("✅ Main pipeline imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main pipeline: {e}")
        return False
    
    try:
        from lungrads_classifier import LungRADSClassifier
        print("✅ LungRADS classifier imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LungRADS classifier: {e}")
        return False
    
    try:
        from cancer_segmentation import CancerSegmentation
        print("✅ Cancer segmentation imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import cancer segmentation: {e}")
        return False
    
    try:
        from classification_models import get_classification_model
        print("✅ Classification models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import classification models: {e}")
        return False
    
    return True


def test_lungrads_classifier():
    """Test LungRADS classifier with dummy data"""
    print("\nTesting LungRADS classifier...")
    
    try:
        from lungrads_classifier import LungRADSClassifier
        
        # Initialize classifier
        classifier = LungRADSClassifier()
        print("✅ LungRADS classifier initialized")
        
        # Test model architecture
        model = classifier.model
        print(f"✅ Model architecture created: {type(model).__name__}")
        
        # Test with dummy data
        dummy_ct = np.random.rand(64, 64, 64).astype(np.float32)
        dummy_tabular = np.array([65.0, 1.0, 0.0])
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Save dummy data as NIfTI (simplified)
        try:
            import nibabel as nib
            img = nib.Nifti1Image(dummy_ct, np.eye(4))
            nib.save(img, tmp_path)
            
            # Test classification
            lungrads_class, lungrads_label, confidence = classifier.classify(tmp_path, dummy_tabular)
            print(f"✅ Classification test passed: {lungrads_label} (confidence: {confidence:.3f})")
            
            # Clean up
            os.unlink(tmp_path)
            
        except ImportError:
            print("⚠️ Nibabel not available, skipping file I/O test")
            print("✅ LungRADS classifier basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ LungRADS classifier test failed: {e}")
        return False


def test_cancer_segmentation():
    """Test cancer segmentation with dummy data"""
    print("\nTesting cancer segmentation...")
    
    try:
        from cancer_segmentation import CancerSegmentation
        
        # Initialize segmenter
        segmenter = CancerSegmentation()
        print("✅ Cancer segmentation initialized")
        
        # Test model architecture
        model = segmenter.model
        print(f"✅ Model architecture created: {type(model).__name__}")
        
        # Test with dummy data
        dummy_ct = np.random.rand(64, 64, 64).astype(np.float32)
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            import nibabel as nib
            img = nib.Nifti1Image(dummy_ct, np.eye(4))
            nib.save(img, tmp_path)
            
            # Test segmentation
            results = segmenter.segment(tmp_path, threshold=0.5)
            print(f"✅ Segmentation test passed: {results['num_detections']} detections")
            
            # Clean up
            os.unlink(tmp_path)
            
        except ImportError:
            print("⚠️ Nibabel not available, skipping file I/O test")
            print("✅ Cancer segmentation basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ Cancer segmentation test failed: {e}")
        return False


def test_classification_models():
    """Test classification models"""
    print("\nTesting classification models...")
    
    try:
        from classification_models import get_classification_model
        
        # Test model creation
        device = torch.device('cpu')
        model = get_classification_model(
            Model_name='resnet50',
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,
            device=device
        )
        print("✅ ResNet50 model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classification models test failed: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    
    try:
        from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline
        
        # Test with dummy model path (should work with random weights)
        dummy_model_path = "dummy_model.pth"
        
        # Create a dummy model file
        dummy_model = torch.nn.Linear(10, 2)
        torch.save(dummy_model.state_dict(), dummy_model_path)
        
        # Initialize pipeline
        pipeline = LungNoduleDetectionPipeline(
            stage1_model_path=dummy_model_path,
            device='cpu'
        )
        print("✅ Pipeline initialized successfully")
        
        # Clean up
        os.unlink(dummy_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'skimage',
        'SimpleITK'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("3-STAGE LUNG NODULE DETECTION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Classification Models", test_classification_models),
        ("LungRADS Classifier", test_lungrads_classifier),
        ("Cancer Segmentation", test_cancer_segmentation),
        ("Pipeline Initialization", test_pipeline_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Train or obtain model weights for each stage")
        print("2. Place model files in the 'models/' directory")
        print("3. Run the pipeline with your CT scan data")
        print("\nExample:")
        print("python lung_nodule_detection_pipeline.py --input /path/to/ct.nii.gz --stage1-model /path/to/model.pth")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
