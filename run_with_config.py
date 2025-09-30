#!/usr/bin/env python3
"""
Run 3-Stage Lung Nodule Detection Pipeline with Configuration File

This script provides a simple way to run the pipeline using a JSON configuration file.

Usage:
    python run_with_config.py --config config.json
    python run_with_config.py --config config.json --validate
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
from lung_nodule_detection_pipeline import LungNoduleDetectionPipeline


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run 3-Stage Lung Nodule Detection Pipeline with Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  python run_with_config.py --config config.json
  
  # Validate configuration without running
  python run_with_config.py --config config.json --validate-only
  
  # Create sample configuration
  python run_with_config.py --create-sample-config
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, don't run pipeline"
    )
    
    parser.add_argument(
        "--create-sample-config",
        action="store_true",
        help="Create a sample configuration file"
    )
    
    args = parser.parse_args()
    
    # Create sample configuration if requested
    if args.create_sample_config:
        from config_loader import create_sample_config
        create_sample_config("sample_config.json")
        print("Sample configuration created: sample_config.json")
        print("Edit this file with your actual paths and use it with --config")
        return
    
    # Validate configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = ConfigLoader(args.config)
        
        # Print configuration summary
        config.print_config_summary()
        
        # Validate paths
        config.print_validation_results()
        
        # Check if only validation requested
        if args.validate_only:
            print("\n✅ Configuration validation completed!")
            return
        
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = LungNoduleDetectionPipeline.from_config(config)
        
        # Get paths from configuration
        ct_file_path = config.get_input_ct_scan()
        output_dir = config.get_output_directory()
        
        # Process CT scan
        print(f"\nProcessing CT scan: {ct_file_path}")
        results = pipeline.process_ct_scan(ct_file_path, output_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Input File: {Path(ct_file_path).name}")
        print(f"Stage 1 - Cancer: {results['stage1_results']['prediction']} ({results['stage1_results']['cancer_probability']:.4f})")
        
        if results['stage1_results']['proceed_to_stage2']:
            print(f"Stage 2 - LungRADS: {results['stage2_results']['lungrads_label']} ({results['stage2_results']['confidence']:.4f})")
            print(f"Stage 3 - Detections: {results['stage3_results']['num_detections']}")
        
        print(f"Overall Success: {'Yes' if results['overall_success'] else 'No'}")
        
        if output_dir:
            print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.config and os.path.exists(args.config):
            print(f"\nTroubleshooting:")
            print(f"1. Check that all paths in {args.config} are correct")
            print(f"2. Run with --validate-only to check paths")
            print(f"3. Create a new config with --create-sample-config")
        sys.exit(1)


if __name__ == "__main__":
    main()
