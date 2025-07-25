#!/usr/bin/env python3
"""
Main Benchmark Runner for Comprehensive Diffusion Model Evaluation

This script provides a unified interface for running all types of benchmarks:
- ML Statistics: Standard machine learning evaluation metrics
- GTO Halo Benchmarking: Domain-specific physical validation
- Complete Evaluation: Both ML statistics and GTO Halo benchmarking
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Reflected-Diffusion'))

from ml_statistics import MLStatisticsBenchmarker, MLStatisticsConfig
from gto_halo_benchmarking import GTOHaloBenchmarker, GTOHaloBenchmarkConfig


def run_ml_statistics_benchmark(args):
    """Run ML statistics benchmark."""
    print("=" * 60)
    print("RUNNING ML STATISTICS BENCHMARK")
    print("=" * 60)
    
    config = MLStatisticsConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sampling_method=args.sampling_method,
        guidance_weight=args.guidance_weight,
        output_dir=f"{args.output_dir}/ml_statistics",
        save_samples=args.save_samples,
        save_plots=args.save_plots,
        device=args.device
    )
    
    benchmarker = MLStatisticsBenchmarker(config)
    results = benchmarker.run_benchmark()
    
    print(f"ML statistics benchmark completed! Results saved to {args.output_dir}/ml_statistics")
    return results


def run_gto_halo_benchmark(args):
    """Run GTO Halo specific benchmark."""
    print("=" * 60)
    print("RUNNING GTO HALO BENCHMARK")
    print("=" * 60)
    
    # Load CR3BP config if provided
    cr3bp_config = None
    if args.cr3bp_config and os.path.exists(args.cr3bp_config):
        import json
        with open(args.cr3bp_config, 'r') as f:
            cr3bp_config = json.load(f)
    
    config = GTOHaloBenchmarkConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sampling_method=args.sampling_method,
        guidance_weight=args.guidance_weight,
        enable_physical_validation=args.enable_physical_validation,
        cr3bp_config=cr3bp_config,
        output_dir=f"{args.output_dir}/gto_halo",
        save_samples=args.save_samples,
        save_plots=args.save_plots,
        device=args.device
    )
    
    benchmarker = GTOHaloBenchmarker(config)
    results = benchmarker.run_benchmark()
    
    print(f"GTO Halo benchmark completed! Results saved to {args.output_dir}/gto_halo")
    return results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Diffusion Model Benchmarking Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both ML statistics and GTO Halo benchmarking
  python run_benchmark.py --benchmark_type both \\
      --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \\
      --data_path data/training_data_boundary_100000.pkl \\
      --num_samples 100

  # Run only ML statistics
  python run_benchmark.py --benchmark_type ml_only \\
      --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \\
      --data_path data/training_data_boundary_100000.pkl \\
      --num_samples 1000

  # Run only GTO Halo benchmarking with CR3BP validation
  python run_benchmark.py --benchmark_type gto_halo_only \\
      --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \\
      --data_path data/training_data_boundary_100000.pkl \\
      --enable_physical_validation \\
      --cr3bp_config cr3bp_validation_config.json

  # Run quick test with minimal samples
  python run_benchmark.py --benchmark_type both \\
      --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \\
      --data_path data/training_data_boundary_100000.pkl \\
      --num_samples 10 \\
      --batch_size 5 \\
      --test_mode
        """
    )
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model directory')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to the model config file or directory (defaults to model_path)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to reference data for comparison')
    
    # Benchmark type
    parser.add_argument('--benchmark_type', type=str, default='both',
                       choices=['ml_only', 'gto_halo_only', 'both'],
                       help='Type of benchmark to run')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for sampling')
    parser.add_argument('--sampling_method', type=str, default='pc',
                       help='Sampling method to use')
    parser.add_argument('--guidance_weight', type=float, default=0.0,
                       help='Classifier-free guidance weight')
    
    # Physical validation
    parser.add_argument('--enable_physical_validation', action='store_true',
                       help='Enable physical validation metrics')
    parser.add_argument('--cr3bp_config', type=str, default='cr3bp_validation_config.json',
                       help='Path to CR3BP validation config file')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--save_samples', action='store_true',
                       help='Save generated samples')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    # Additional options
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with minimal samples')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set config_path to model_path if not provided
    if args.config_path is None:
        args.config_path = args.model_path
    
    # Set device
    if args.device == 'auto':
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Test mode adjustments
    if args.test_mode:
        args.num_samples = min(args.num_samples, 10)
        args.batch_size = min(args.batch_size, 5)
        print("Running in test mode with reduced samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    results = {}
    
    if args.benchmark_type in ['ml_only', 'both']:
        try:
            ml_results = run_ml_statistics_benchmark(args)
            results['ml_statistics'] = ml_results
        except Exception as e:
            print(f"Error in ML statistics benchmark: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    if args.benchmark_type in ['gto_halo_only', 'both']:
        try:
            gto_halo_results = run_gto_halo_benchmark(args)
            results['gto_halo'] = gto_halo_results
        except Exception as e:
            print(f"Error in GTO Halo benchmark: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for benchmark_type, result in results.items():
        print(f"\n{benchmark_type.upper()} BENCHMARK:")
        if 'sampling_efficiency' in result:
            efficiency = result['sampling_efficiency']
            print(f"  Total sampling time: {efficiency.get('total_sampling_time', 'N/A'):.2f}s")
            print(f"  Samples per second: {efficiency.get('samples_per_second', 'N/A'):.2f}")
        
        if 'gto_halo_metrics' in result:
            metrics = result['gto_halo_metrics']
            # Note: No boundary violation rate for reflected diffusion model
            print(f"  Class label mean: {metrics.get('class_label_mean', 'N/A'):.6f}")
        
        if 'standard_metrics' in result:
            metrics = result['standard_metrics']
            print(f"  MSE: {metrics.get('mse', 'N/A'):.6f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Check the summary.txt files in each subdirectory for detailed results.")
    
    return results


if __name__ == "__main__":
    main() 