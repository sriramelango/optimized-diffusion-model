#!/usr/bin/env python3
"""
Fast Multithreaded GTO Halo Benchmarking Runner

This script provides an easy interface to run the fast multithreaded GTO Halo benchmarking
with pre-warmed threads to reduce initialization delays.

Example usage:
python run_fast_benchmark.py \
    --model_path "Training Runs/2025.07.18_031529" \
    --data_path "GTO_Halo_DM/data/training_data_boundary_100000.pkl" \
    --num_samples 8 \
    --max_workers 8 \
    --output_dir "benchmark_results/gto_halo_8_samples_fast"
"""

import os
import sys
import argparse
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gto_halo_benchmarking_multithreaded_fast import GTOHaloBenchmarker, GTOHaloBenchmarkConfig

def main():
    parser = argparse.ArgumentParser(description='Fast Multithreaded GTO Halo Benchmarking')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint directory')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to reference data')
    
    # Optional arguments
    parser.add_argument('--num_samples', type=int, default=8, 
                       help='Number of samples to generate (default: 8)')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for sampling (default: 4)')
    parser.add_argument('--max_workers', type=int, default=8, 
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--chunk_size', type=int, default=1, 
                       help='Number of samples per thread (default: 1)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--enable_physical_validation', action='store_true', default=True, 
                       help='Enable physical validation (default: True)')
    parser.add_argument('--pre_warm_threads', action='store_true', default=True, 
                       help='Pre-warm threads to reduce initialization delays (default: True)')
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"benchmark_results/gto_halo_{args.num_samples}_samples_fast_{timestamp}"
    
    print("=" * 80)
    print("FAST MULTITHREADED GTO HALO BENCHMARKING")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Physical validation: {args.enable_physical_validation}")
    print(f"Pre-warm threads: {args.pre_warm_threads}")
    print("=" * 80)
    
    # Create configuration
    config = GTOHaloBenchmarkConfig(
        model_path=args.model_path,
        config_path=args.model_path,  # Use model_path as config_path
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        pre_warm_threads=args.pre_warm_threads,
        output_dir=args.output_dir,
        enable_physical_validation=args.enable_physical_validation
    )
    
    # Run benchmark
    benchmarker = GTOHaloBenchmarker(config)
    results = benchmarker.run_benchmark()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    print(f"Total sampling time: {results['sampling_efficiency_metrics'].get('total_sampling_time', 0):.2f}s")
    print(f"Samples per second: {results['sampling_efficiency_metrics'].get('samples_per_second', 0):.3f}")
    print(f"Class label mean: {results['gto_halo_metrics'].get('class_label_mean', 0):.6f}")
    print(f"Feasible ratio: {results['physical_validation_metrics'].get('feasible_ratio', 0):.3f}")
    print(f"Local optimal ratio: {results['physical_validation_metrics'].get('local_optimal_ratio', 0):.3f}")
    print(f"Results saved to: {args.output_dir}")
    print("Check the summary.txt files in each subdirectory for detailed results.")

if __name__ == "__main__":
    main() 