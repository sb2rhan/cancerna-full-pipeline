#!/usr/bin/env python3
"""
Lung Cancer Segmentation System

This module provides lung cancer segmentation functionality with coordinate
and diameter extraction based on the FAUNet architecture from the notebook.

Usage:
    from cancer_segmentation import CancerSegmentation
    
    segmenter = CancerSegmentation()
    results = segmenter.segment(ct_file_path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from skimage import measure, morphology
import matplotlib.pyplot as plt


class CancerSegmentation:
    """Lung Cancer Segmentation System"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize cancer segmentation system
        
        Args:
            model_path: Path to trained model (optional)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model_path = model_path
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the cancer segmentation model"""
        print("Loading Cancer Segmentation Model...")
        
        # Create FAUNet model
        self.model = self._create_faunet_model()
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
                print("✅ Cancer segmentation model loaded successfully!")
                
            except Exception as e:
                print(f"⚠️ Error loading segmentation model: {e}")
                print("Using random weights...")
        else:
            print("⚠️ No trained segmentation model found, using random weights")
        
        self.model.eval()
    
    def _create_faunet_model(self):
        """Create FAUNet model architecture based on the notebook"""
        
        class GaussianMembershipFunction(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.mu = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.1)
                self.sigma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.8 + torch.randn(1, channels, 1, 1) * 0.2)
                self.m = nn.Parameter(torch.tensor(2.0))
                
            def forward(self, x):
                sigma = torch.clamp(self.sigma, min=0.1, max=5.0)
                m = torch.clamp(self.m, min=1.0, max=4.0)
                
                diff = torch.abs((x - self.mu) / (sigma + 1e-8))
                p = diff ** m
                return torch.exp(-0.5 * p)

        class AttentionGate(nn.Module):
            def __init__(self, F_g, F_l, F_int):
                super().__init__()
                self.W_g = nn.Sequential(
                    nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(F_int),
                )
                self.W_x = nn.Sequential(
                    nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(F_int),
                )
                self.psi = nn.Sequential(
                    nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1),
                )
                self.gmf = GaussianMembershipFunction(1)
                self.dropout = nn.Dropout2d(0.1)
                
            def forward(self, x, g):
                if g.size()[2:] != x.size()[2:]:
                    g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
                    
                g1 = self.W_g(g)
                x1 = self.W_x(x)
                psi = F.relu(g1 + x1)
                psi = self.dropout(psi)
                psi = self.psi(psi)
                psi = self.gmf(psi)
                return x * psi

        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                
                self.skip = nn.Sequential()
                if in_ch != out_ch:
                    self.skip = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch)
                    )
                
                self.dropout = nn.Dropout2d(0.1)
                
            def forward(self, x):
                residual = self.skip(x)
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.dropout(out)
                out = self.bn2(self.conv2(out))
                out += residual
                return F.relu(out)

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                ResidualBlock(in_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            )

        class PyramidPoolingModule(nn.Module):
            def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
                super().__init__()
                self.stages = nn.ModuleList([
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                        nn.BatchNorm2d(in_channels // len(pool_sizes)),
                        nn.ReLU(inplace=True)
                    ) for pool_size in pool_sizes
                ])
                
            def forward(self, x):
                h, w = x.size(2), x.size(3)
                pyramids = [x]
                
                for stage in self.stages:
                    pyramid = stage(x)
                    pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=False)
                    pyramids.append(pyramid)
                    
                return torch.cat(pyramids, dim=1)

        class FAUNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1):
                super().__init__()
                self.enc1 = conv_block(in_channels, 64)
                self.enc2 = conv_block(64, 128)
                self.enc3 = conv_block(128, 256)
                self.enc4 = conv_block(256, 512)

                self.pool = nn.MaxPool2d(2, 2)
                self.bottleneck = conv_block(512, 1024)
                self.ppm = PyramidPoolingModule(1024)
                self.bottleneck_final = nn.Sequential(
                    nn.Conv2d(1024 + 1024, 1024, 3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.2)
                )
                
                self.gmf = GaussianMembershipFunction(1024)

                self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
                self.dec4 = conv_block(1024, 512)

                self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
                self.dec3 = conv_block(512, 256)

                self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
                self.dec2 = conv_block(256, 128)

                self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
                self.dec1 = conv_block(128, 64)

                self.final_conv = nn.Sequential(
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(32, out_channels, 1)
                )

                self._initialize_weights()

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

                b = self.pool(e4)
                b = self.bottleneck(b)
                b = self.ppm(b)
                b = self.bottleneck_final(b)
                b = self.gmf(b)

                d4 = self.up4(b)
                e4 = self.att4(e4, d4)
                d4 = self.dec4(torch.cat([d4, e4], dim=1))

                d3 = self.up3(d4)
                e3 = self.att3(e3, d3)
                d3 = self.dec3(torch.cat([d3, e3], dim=1))

                d2 = self.up2(d3)
                e2 = self.att2(e2, d2)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))

                d1 = self.up1(d2)
                e1 = self.att1(e1, d1)
                d1 = self.dec1(torch.cat([d1, e1], dim=1))

                out = self.final_conv(d1)
                return torch.sigmoid(out)
        
        return FAUNet(in_channels=1, out_channels=1)
    
    def segment(self, ct_file_path: str, threshold: float = 0.5) -> Dict:
        """
        Segment lung cancer and extract coordinates and diameter
        
        Args:
            ct_file_path: Path to CT scan file
            threshold: Segmentation threshold (0-1)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Load CT scan
            ct_image = sitk.ReadImage(ct_file_path)
            ct_array = sitk.GetArrayFromImage(ct_image)
            spacing = np.array(ct_image.GetSpacing())
            origin = np.array(ct_image.GetOrigin())
            
            # Process slices
            segmentation_results = []
            
            # Process all slices in the volume
            slice_range = range(0, ct_array.shape[0])
            
            print(f"Processing {len(slice_range)} slices...")
            
            for slice_idx in slice_range:
                slice_data = ct_array[slice_idx]
                
                # Normalize slice
                slice_normalized = self._normalize_slice(slice_data)
                
                # Resize to model input size
                slice_resized = self._resize_slice(slice_normalized, (512, 512))
                
                # Convert to tensor
                slice_tensor = torch.FloatTensor(slice_resized).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Run segmentation
                with torch.no_grad():
                    pred_mask = self.model(slice_tensor)
                    pred_mask = pred_mask.squeeze().cpu().numpy()
                
                # Find connected components
                binary_mask = (pred_mask > threshold).astype(np.uint8)
                
                # Apply morphological operations to clean up
                binary_mask = morphology.remove_small_objects(binary_mask, min_size=10)
                binary_mask = morphology.binary_closing(binary_mask, morphology.disk(2))
                
                labeled_mask, num_features = measure.label(binary_mask, return_num=True)
                
                # Analyze each component
                for region in measure.regionprops(labeled_mask):
                    if region.area > 20:  # Filter small regions
                        # Calculate coordinates (center of mass)
                        center_y, center_x = region.centroid
                        
                        # Calculate diameter in mm
                        diameter_pixels = region.equivalent_diameter
                        diameter_mm = diameter_pixels * spacing[1]  # Assuming square pixels
                        
                        # Convert to world coordinates
                        world_coords = self._pixel_to_world_coords(
                            center_x, center_y, slice_idx, ct_image
                        )
                        
                        # Calculate confidence (max probability in region)
                        region_mask = (labeled_mask == region.label)
                        confidence = float(np.max(pred_mask[region_mask]))
                        
                        segmentation_results.append({
                            'slice_index': slice_idx,
                            'center_pixel': (int(center_x), int(center_y)),
                            'center_world': world_coords,
                            'diameter_mm': diameter_mm,
                            'area_pixels': region.area,
                            'confidence': confidence,
                            'bbox': region.bbox  # (min_row, min_col, max_row, max_col)
                        })
            
            # Sort by confidence and return top results
            segmentation_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Remove overlapping detections
            filtered_results = self._remove_overlapping_detections(segmentation_results)
            
            return {
                'num_detections': len(filtered_results),
                'detections': filtered_results[:10],  # Top 10 detections
                'success': True,
                'threshold_used': threshold
            }
            
        except Exception as e:
            print(f"Error in cancer segmentation: {e}")
            return {
                'num_detections': 0,
                'detections': [],
                'success': False,
                'error': str(e)
            }
    
    def _normalize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """Normalize slice to [0, 1] range"""
        # Clip to reasonable HU range
        slice_data = np.clip(slice_data, -1000, 400)
        # Normalize to [0, 1]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        return slice_data.astype(np.float32)
    
    def _resize_slice(self, slice_data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize slice to target size"""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_size[0] / slice_data.shape[0],
            target_size[1] / slice_data.shape[1]
        ]
        
        return zoom(slice_data, zoom_factors, order=1)
    
    def _pixel_to_world_coords(self, x: float, y: float, z: int, ct_image) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates"""
        origin = np.array(ct_image.GetOrigin())
        spacing = np.array(ct_image.GetSpacing())
        
        world_x = origin[0] + x * spacing[0]
        world_y = origin[1] + y * spacing[1]
        world_z = origin[2] + z * spacing[2]
        
        return (world_x, world_y, world_z)
    
    def _remove_overlapping_detections(self, detections: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping detections based on spatial proximity"""
        if not detections:
            return detections
        
        filtered = []
        
        for detection in detections:
            is_overlapping = False
            
            for existing in filtered:
                # Calculate distance between detections
                dist = np.sqrt(
                    (detection['center_pixel'][0] - existing['center_pixel'][0])**2 +
                    (detection['center_pixel'][1] - existing['center_pixel'][1])**2
                )
                
                # Calculate average diameter
                avg_diameter = (detection['diameter_mm'] + existing['diameter_mm']) / 2
                
                # Check if overlapping
                if dist < avg_diameter * overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(detection)
        
        return filtered
    
    def visualize_results(self, ct_file_path: str, results: Dict, save_path: Optional[str] = None):
        """Visualize segmentation results"""
        try:
            # Load CT scan
            ct_image = sitk.ReadImage(ct_file_path)
            ct_array = sitk.GetArrayFromImage(ct_image)
            
            # Get slices with detections
            detection_slices = set()
            for detection in results['detections']:
                detection_slices.add(detection['slice_index'])
            
            # Create visualization
            num_slices = min(len(detection_slices), 6)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, slice_idx in enumerate(sorted(detection_slices)[:num_slices]):
                if i >= 6:
                    break
                
                # Get slice data
                slice_data = ct_array[slice_idx]
                
                # Normalize for display
                slice_display = np.clip(slice_data, -1000, 400)
                slice_display = (slice_display - slice_display.min()) / (slice_display.max() - slice_display.min())
                
                # Display slice
                axes[i].imshow(slice_display, cmap='gray')
                
                # Overlay detections
                for detection in results['detections']:
                    if detection['slice_index'] == slice_idx:
                        center = detection['center_pixel']
                        diameter = detection['diameter_mm']
                        
                        # Draw circle
                        circle = plt.Circle(center, diameter/2, color='red', fill=False, linewidth=2)
                        axes[i].add_patch(circle)
                        
                        # Add text
                        axes[i].text(center[0], center[1], f"{diameter:.1f}mm", 
                                   color='red', fontsize=8, ha='center', va='center')
                
                axes[i].set_title(f'Slice {slice_idx}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_slices, 6):
                axes[i].axis('off')
            
            plt.suptitle(f'Cancer Segmentation Results - {results["num_detections"]} detections', fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")


def main():
    """Test the cancer segmentation system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cancer Segmentation System")
    parser.add_argument("--input", "-i", required=True, help="Path to CT scan file")
    parser.add_argument("--model", "-m", help="Path to trained model file")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--output", "-o", help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = CancerSegmentation(model_path=args.model)
    
    # Run segmentation
    print(f"Segmenting CT scan: {args.input}")
    results = segmenter.segment(args.input, threshold=args.threshold)
    
    print(f"\nResults:")
    print(f"Number of detections: {results['num_detections']}")
    print(f"Success: {results['success']}")
    
    if results['detections']:
        print(f"\nTop detections:")
        for i, detection in enumerate(results['detections'][:5]):
            print(f"  {i+1}. Slice {detection['slice_index']}: Center={detection['center_world']}, "
                  f"Diameter={detection['diameter_mm']:.2f}mm, Confidence={detection['confidence']:.4f}")
    
    # Create visualization
    if results['success'] and results['detections']:
        segmenter.visualize_results(args.input, results, args.output)


if __name__ == "__main__":
    main()
