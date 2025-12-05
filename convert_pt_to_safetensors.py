#!/usr/bin/env python3
"""
Convert PyTorch .pt model files to .safetensors format.

This script converts all .pt files in the saved_models directory to the
safetensors format, which is safer and faster than pickle-based formats.

Usage:
    python convert_pt_to_safetensors.py
    
    or to convert a specific model:
    python convert_pt_to_safetensors.py --model model_name.pt
"""

import argparse
import os
from pathlib import Path
import torch
from safetensors.torch import save_file


def convert_pt_to_safetensors(pt_path, safetensors_path=None):
    """
    Convert a .pt model file to .safetensors format.
    
    Args:
        pt_path (str or Path): Path to the .pt file
        safetensors_path (str or Path, optional): Output path for .safetensors file.
            If None, will use the same name as the input file with .safetensors extension.
    
    Returns:
        Path: Path to the created .safetensors file
    """
    pt_path = Path(pt_path)
    
    if not pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {pt_path}")
    
    if safetensors_path is None:
        safetensors_path = pt_path.with_suffix('.safetensors')
    else:
        safetensors_path = Path(safetensors_path)
    
    print(f"Loading model from: {pt_path}")
    
    # Load the .pt file
    try:
        state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        # Handle different possible formats
        if isinstance(state_dict, dict):
            # If the loaded object has a 'state_dict' key, use that
            if 'state_dict' in state_dict:
                tensors = state_dict['state_dict']
                print(f"  Found 'state_dict' key in checkpoint")
            # If it has 'model' key
            elif 'model' in state_dict:
                tensors = state_dict['model']
                print(f"  Found 'model' key in checkpoint")
            # Otherwise assume it's already a state dict
            else:
                tensors = state_dict
                print(f"  Using entire dictionary as state dict")
        else:
            raise ValueError(f"Unexpected model format: {type(state_dict)}")
        
        # Ensure all values are tensors
        tensor_dict = {}
        for key, value in tensors.items():
            if isinstance(value, torch.Tensor):
                tensor_dict[key] = value
            else:
                print(f"  Warning: Skipping non-tensor key '{key}' with type {type(value)}")
        
        print(f"  Found {len(tensor_dict)} tensors")
        
        # Save as safetensors
        print(f"Saving to: {safetensors_path}")
        save_file(tensor_dict, safetensors_path)
        
        # Verify file size
        pt_size = pt_path.stat().st_size / (1024 * 1024)  # MB
        st_size = safetensors_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Original .pt file: {pt_size:.2f} MB")
        print(f"  New .safetensors file: {st_size:.2f} MB")
        print(f"✓ Conversion successful!\n")
        
        return safetensors_path
        
    except Exception as e:
        print(f"✗ Error converting {pt_path}: {e}\n")
        raise


def convert_all_models_in_directory(directory):
    """
    Convert all .pt files in a directory to .safetensors format.
    
    Args:
        directory (str or Path): Directory containing .pt files
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all .pt files
    pt_files = list(directory.glob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {directory}")
        return
    
    print(f"Found {len(pt_files)} .pt file(s) in {directory}\n")
    print("=" * 70)
    
    successful = []
    failed = []
    
    for pt_file in pt_files:
        try:
            safetensors_path = convert_pt_to_safetensors(pt_file)
            successful.append((pt_file, safetensors_path))
        except Exception as e:
            failed.append((pt_file, str(e)))
    
    # Summary
    print("=" * 70)
    print("\nConversion Summary:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successfully converted:")
        for pt_file, st_file in successful:
            print(f"    {pt_file.name} → {st_file.name}")
    
    if failed:
        print("\n✗ Failed conversions:")
        for pt_file, error in failed:
            print(f"    {pt_file.name}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pt model files to .safetensors format"
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model file to convert (if not provided, converts all models in saved_models/)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='saved_models',
        help='Input directory containing .pt files (default: saved_models)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for .safetensors file (only used with --model)'
    )
    
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    
    if args.model:
        # Convert a specific model
        model_path = Path(args.model)
        if not model_path.is_absolute():
            # Try relative to input_dir first
            if (input_dir / model_path).exists():
                model_path = input_dir / model_path
            # Then try relative to current directory
            elif not model_path.exists():
                print(f"Error: Model file not found: {args.model}")
                return 1
        
        output_path = args.output if args.output else None
        convert_pt_to_safetensors(model_path, output_path)
    else:
        # Convert all models in directory
        convert_all_models_in_directory(input_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())
