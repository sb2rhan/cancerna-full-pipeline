#!/usr/bin/env python3
"""
LungRADS Classification System

This module provides LungRADS classification functionality for lung nodules.
LungRADS categories: 1, 2, 3, 4A, 4B, 4X

Usage:
    from lungrads_classifier import LungRADSClassifier
    
    classifier = LungRADSClassifier()
    lungrads_class, lungrads_label, confidence = classifier.classify(ct_file_path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class LungRADSClassifier:
    """LungRADS Classification System"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 encoder_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 label_classes_path: Optional[str] = None,
                 encoders_and_scalers_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model (optional)
            encoder_path: Path to encoder pickle file (optional)
            scaler_path: Path to scaler pickle file (optional)
            label_classes_path: Path to label classes pickle file (optional)
            encoders_and_scalers_path: Path to combined encoders and scalers pickle file (optional)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.label_classes_path = label_classes_path
        self.encoders_and_scalers_path = encoders_and_scalers_path
        self.model = None
        self.encoder = None
        self.scaler = None
        self.label_classes = None
        
        # Default LungRADS class mapping (will be overridden if label_classes_path provided)
        self.lungrads_labels = {
            0: "1",
            1: "2",
            2: "3",
            3: "4A",
            4: "4B",
            5: "4X"
        }
        
        self._load_components()
    
    def _load_components(self):
        """Load all components: model, encoder, scaler, and label classes"""
        print("Loading LungRADS Classification Components...")
        
        # Load label classes
        self._load_label_classes()
        
        # Load encoders and scalers (combined file takes priority)
        if self.encoders_and_scalers_path:
            self._load_encoders_and_scalers()
        else:
            # Load encoder and scaler separately
            self._load_encoder()
            self._load_scaler()
        
        # Load model
        self._load_model()
    
    def _load_label_classes(self):
        """Load label classes from pickle file"""
        if self.label_classes_path and Path(self.label_classes_path).exists():
            try:
                import pickle
                with open(self.label_classes_path, 'rb') as f:
                    self.label_classes = pickle.load(f)
                print("✅ Label classes loaded successfully!")
                
                # Update lungrads_labels mapping
                if isinstance(self.label_classes, dict):
                    print(f"   Loaded label classes (dict): {self.label_classes}")
                    # Check if it contains 'lr_classes' key
                    if 'lr_classes' in self.label_classes:
                        label_list = self.label_classes['lr_classes']
                        print(f"   Extracted lr_classes: {label_list}")
                        self.lungrads_labels = {i: str(label) for i, label in enumerate(label_list)}
                    else:
                        # Treat the dict as direct mapping
                        self.lungrads_labels = {int(k): str(v) for k, v in self.label_classes.items() if str(k).isdigit()}
                    print(f"   Final lungrads_labels mapping: {self.lungrads_labels}")
                elif isinstance(self.label_classes, list):
                    print(f"   Loaded label classes (list): {self.label_classes}")
                    self.lungrads_labels = {i: str(label) for i, label in enumerate(self.label_classes)}
                    print(f"   Final lungrads_labels mapping: {self.lungrads_labels}")
                else:
                    print("⚠️ Unexpected label classes format, using default mapping")
                    
            except Exception as e:
                print(f"⚠️ Error loading label classes: {e}")
                print("Using default LungRADS mapping")
        else:
            print("⚠️ No label classes file found, using default mapping")
    
    def _load_encoder(self):
        """Load encoder from pickle file"""
        if self.encoder_path and Path(self.encoder_path).exists():
            try:
                import pickle
                with open(self.encoder_path, 'rb') as f:
                    encoder_data = pickle.load(f)
                
                # Handle different formats
                if isinstance(encoder_data, dict):
                    # If it's a dictionary, look for encoder key
                    if 'encoder' in encoder_data:
                        self.encoder = encoder_data['encoder']
                    elif 'encoders' in encoder_data:
                        self.encoder = encoder_data['encoders']
                    else:
                        # Assume the whole dict is encoder data
                        self.encoder = encoder_data
                else:
                    # Assume it's directly the encoder
                    self.encoder = encoder_data
                
                print("✅ Encoder loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading encoder: {e}")
                print("Using default preprocessing")
        else:
            print("⚠️ No encoder file found, using default preprocessing")
    
    def _load_scaler(self):
        """Load scaler from pickle file"""
        if self.scaler_path and Path(self.scaler_path).exists():
            try:
                import pickle
                with open(self.scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                
                # Handle different formats
                if isinstance(scaler_data, dict):
                    # If it's a dictionary, look for scaler key
                    if 'scaler' in scaler_data:
                        self.scaler = scaler_data['scaler']
                    elif 'scalers' in scaler_data:
                        self.scaler = scaler_data['scalers']
                    else:
                        # Assume the whole dict is scaler data
                        self.scaler = scaler_data
                else:
                    # Assume it's directly the scaler
                    self.scaler = scaler_data
                
                print("✅ Scaler loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading scaler: {e}")
                print("Using default scaling")
        else:
            print("⚠️ No scaler file found, using default scaling")
    
    def _load_encoders_and_scalers(self):
        """Load encoders and scalers from combined pickle file"""
        if self.encoders_and_scalers_path and Path(self.encoders_and_scalers_path).exists():
            try:
                import pickle
                with open(self.encoders_and_scalers_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"✅ Combined encoders and scalers file loaded successfully!")
                print(f"File contains: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                # Handle different formats
                if isinstance(data, dict):
                    # Look for common keys
                    if 'encoder' in data:
                        self.encoder = data['encoder']
                        print("✅ Encoder extracted from combined file")
                    elif 'encoders' in data:
                        self.encoder = data['encoders']
                        print("✅ Encoders extracted from combined file")
                    
                    if 'scaler' in data:
                        self.scaler = data['scaler']
                        print("✅ Scaler extracted from combined file")
                    elif 'scalers' in data:
                        self.scaler = data['scalers']
                        print("✅ Scalers extracted from combined file")
                    
                    # If no specific keys found, try to infer from content
                    if not self.encoder and not self.scaler:
                        print("⚠️ Could not find encoder/scaler keys in combined file")
                        print("Available keys:", list(data.keys()))
                        # You might need to adjust this based on your file structure
                        # For example, if your file has different key names
                        
                else:
                    print("⚠️ Combined file is not a dictionary, cannot extract encoder/scaler")
                    
            except Exception as e:
                print(f"⚠️ Error loading combined encoders and scalers: {e}")
                print("Falling back to separate encoder/scaler files")
                # Fall back to separate files
                self._load_encoder()
                self._load_scaler()
        else:
            print("⚠️ No combined encoders and scalers file found")
            # Fall back to separate files
            self._load_encoder()
            self._load_scaler()
    
    def _load_model(self):
        """Load the LungRADS classification model"""
        print("Loading LungRADS Classification Model...")
        
        # Create model architecture
        self.model = self._create_lungrads_model()
        self.model.to(self.device)
        
        # Load trained weights if available
        if self.model_path and Path(self.model_path).exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
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
                
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ LungRADS model loaded successfully!")
                
            except Exception as e:
                print(f"⚠️ Error loading LungRADS model: {e}")
                print("Using random weights...")
        else:
            print("⚠️ No trained LungRADS model found, using random weights")
        
        self.model.eval()
    
    def _create_lungrads_model(self):
        """Create LungRADS classification model architecture"""
        
        class LungRADS3DNet(nn.Module):
            def __init__(self, n_classes=6, tab_dim=3):
                super().__init__()
                
                # 3D CNN backbone for CT scan analysis
                self.ct_backbone = nn.Sequential(
                    # First block
                    nn.Conv3d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),  # /2
                    
                    # Second block
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),  # /4
                    
                    # Third block
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),  # /8
                    
                    # Fourth block
                    nn.Conv3d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm3d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d(1)  # Global average pooling
                )
                
                # Tabular features processing
                self.tabular_branch = nn.Sequential(
                    nn.Linear(tab_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # Feature fusion and classification
                self.fusion = nn.Sequential(
                    nn.Linear(256 + 32, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, n_classes)
                )
                
            def forward(self, ct_volume, tabular_features):
                # Process CT volume
                ct_features = self.ct_backbone(ct_volume)
                ct_features = ct_features.view(ct_features.size(0), -1)  # Flatten
                
                # Process tabular features
                tab_features = self.tabular_branch(tabular_features)
                
                # Fuse features
                combined_features = torch.cat([ct_features, tab_features], dim=1)
                
                # Classification
                output = self.fusion(combined_features)
                
                return output
        
        return LungRADS3DNet(n_classes=6, tab_dim=3)
    
    def classify(self, ct_file_path: str, tabular_features: Optional[np.ndarray] = None) -> Tuple[int, str, float]:
        """
        Classify CT scan for LungRADS category
        
        Args:
            ct_file_path: Path to CT scan file
            tabular_features: Optional tabular features [age, gender, smoking_status]
            
        Returns:
            Tuple of (lungrads_class, lungrads_label, confidence)
        """
        try:
            # Load and preprocess CT scan
            ct_volume, spacing = self._load_and_preprocess_ct(ct_file_path)
            
            # Create tabular features if not provided
            if tabular_features is None:
                tabular_features = np.array([65.0, 1.0, 0.0])  # [age, gender, smoking_status]
            
            # Apply encoder and scaler if available
            tabular_features = self._preprocess_tabular_features(tabular_features)
            
            # Convert to tensors
            ct_tensor = torch.FloatTensor(ct_volume).unsqueeze(0).unsqueeze(0).to(self.device)
            tab_tensor = torch.FloatTensor(tabular_features).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(ct_tensor, tab_tensor)
                probabilities = F.softmax(output, dim=1)
                
                # Get predicted class
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Map to LungRADS label
                if predicted_class in self.lungrads_labels:
                    lungrads_label = self.lungrads_labels[predicted_class]
                else:
                    print(f"   Warning: Predicted class {predicted_class} not found in lungrads_labels")
                    print(f"   Available keys: {list(self.lungrads_labels.keys())}")
                    lungrads_label = str(predicted_class)  # Fallback to class number
                
                return predicted_class, lungrads_label, confidence
                
        except Exception as e:
            print(f"Error in LungRADS classification: {e}")
            return 0, "1", 0.0
    
    def _preprocess_tabular_features(self, tabular_features: np.ndarray) -> np.ndarray:
        """Preprocess tabular features using encoder and scaler if available"""
        try:
            # Apply encoder if available
            if self.encoder is not None:
                # Assuming encoder expects 2D input
                tabular_features = tabular_features.reshape(1, -1)
                tabular_features = self.encoder.transform(tabular_features)
                tabular_features = tabular_features.flatten()
            
            # Apply scaler if available
            if self.scaler is not None:
                # Assuming scaler expects 2D input
                tabular_features = tabular_features.reshape(1, -1)
                tabular_features = self.scaler.transform(tabular_features)
                tabular_features = tabular_features.flatten()
            
            return tabular_features
            
        except Exception as e:
            print(f"Warning: Error in tabular feature preprocessing: {e}")
            print("Using original features")
            return tabular_features
    
    def _load_and_preprocess_ct(self, ct_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess CT scan"""
        # Load CT scan
        ct_image = sitk.ReadImage(ct_file_path)
        ct_array = sitk.GetArrayFromImage(ct_image)
        spacing = np.array(ct_image.GetSpacing())
        
        # Normalize CT array
        ct_array = self._normalize_ct(ct_array)
        
        # Resize to expected input size
        target_size = (64, 64, 64)
        ct_array = self._resize_3d(ct_array, target_size)
        
        return ct_array, spacing

    def _normalize_ct(self, ct_array: np.ndarray) -> np.ndarray:
        """Normalize CT array to [0, 1] range"""
        # Clip to reasonable HU range
        ct_array = np.clip(ct_array, -1000, 400)
        # Normalize to [0, 1]
        ct_array = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min())
        return ct_array.astype(np.float32)

    def _resize_3d(self, array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D array to target size"""
        from scipy.ndimage import zoom

        zoom_factors = [
            target_size[0] / array.shape[0],
            target_size[1] / array.shape[1],
            target_size[2] / array.shape[2]
        ]

        return zoom(array, zoom_factors, order=1)

    def get_class_probabilities(self, ct_file_path: str, tabular_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get probability distribution over all LungRADS classes
        
        Args:
            ct_file_path: Path to CT scan file
            tabular_features: Optional tabular features
            
        Returns:
            Dictionary mapping LungRADS labels to probabilities
        """
        try:
            # Load and preprocess CT scan
            ct_volume, _ = self._load_and_preprocess_ct(ct_file_path)
            
            # Create tabular features if not provided
            if tabular_features is None:
                tabular_features = np.array([65.0, 1.0, 0.0])
            
            # Convert to tensors
            ct_tensor = torch.FloatTensor(ct_volume).unsqueeze(0).unsqueeze(0).to(self.device)
            tab_tensor = torch.FloatTensor(tabular_features).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(ct_tensor, tab_tensor)
                probabilities = F.softmax(output, dim=1)
                
                # Convert to dictionary
                prob_dict = {}
                for i, (class_id, label) in enumerate(self.lungrads_labels.items()):
                    prob_dict[label] = probabilities[0, class_id].item()
                
                return prob_dict
                
        except Exception as e:
            print(f"Error getting class probabilities: {e}")
            return {label: 0.0 for label in self.lungrads_labels.values()}
    
    def explain_classification(self, ct_file_path: str, tabular_features: Optional[np.ndarray] = None) -> Dict:
        """
        Provide explanation for LungRADS classification
        
        Args:
            ct_file_path: Path to CT scan file
            tabular_features: Optional tabular features
            
        Returns:
            Dictionary with classification explanation
        """
        # Get classification result
        lungrads_class, lungrads_label, confidence = self.classify(ct_file_path, tabular_features)
        
        # Get all class probabilities
        probabilities = self.get_class_probabilities(ct_file_path, tabular_features)
        
        # Create explanation
        explanation = {
            'predicted_class': lungrads_class,
            'predicted_label': lungrads_label,
            'confidence': confidence,
            'all_probabilities': probabilities,
            'interpretation': self._get_interpretation(lungrads_label, confidence)
        }
        
        return explanation
    
    def _get_interpretation(self, lungrads_label: str, confidence: float) -> str:
        """Get human-readable interpretation of LungRADS classification"""
        interpretations = {
            "1": "Benign nodule - No follow-up needed",
            "2": "Probably benign - 6-month follow-up recommended",
            "3": "Probably malignant - 3-month follow-up recommended",
            "4A": "Suspicious - 1-month follow-up or biopsy recommended",
            "4B": "Suspicious - Biopsy recommended",
            "4X": "Suspicious with additional features - Immediate evaluation recommended"
        }
        
        base_interpretation = interpretations.get(lungrads_label, "Unknown classification")
        
        if confidence > 0.8:
            confidence_level = "High confidence"
        elif confidence > 0.6:
            confidence_level = "Moderate confidence"
        else:
            confidence_level = "Low confidence"
        
        return f"{base_interpretation} ({confidence_level}: {confidence:.2f})"


def main():
    """Test the LungRADS classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LungRADS Classifier")
    parser.add_argument("--input", "-i", required=True, help="Path to CT scan file")
    parser.add_argument("--model", "-m", help="Path to trained model file")
    parser.add_argument("--age", type=float, default=65.0, help="Patient age")
    parser.add_argument("--gender", type=float, default=1.0, help="Patient gender (1=male, 0=female)")
    parser.add_argument("--smoking", type=float, default=0.0, help="Smoking status (1=smoker, 0=non-smoker)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = LungRADSClassifier(model_path=args.model)
    
    # Prepare tabular features
    tabular_features = np.array([args.age, args.gender, args.smoking])
    
    # Run classification
    print(f"Classifying CT scan: {args.input}")
    print(f"Patient info: Age={args.age}, Gender={'Male' if args.gender else 'Female'}, Smoking={'Yes' if args.smoking else 'No'}")
    
    lungrads_class, lungrads_label, confidence = classifier.classify(args.input, tabular_features)
    
    print(f"\nResults:")
    print(f"LungRADS Class: {lungrads_class}")
    print(f"LungRADS Label: {lungrads_label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Get detailed explanation
    explanation = classifier.explain_classification(args.input, tabular_features)
    print(f"\nInterpretation: {explanation['interpretation']}")
    
    print(f"\nAll class probabilities:")
    for label, prob in explanation['all_probabilities'].items():
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
