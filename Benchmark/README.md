# Comprehensive Diffusion Model Benchmarking Framework

This framework provides comprehensive benchmarking for diffusion models applied to GTO Halo trajectory optimization, including both standard ML metrics and domain-specific physical validation.

## Quick Start

```bash
# Run complete benchmark with available checkpoints
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 1000 \
    --benchmark_type both \
    --enable_physical_validation

# Run only ML statistics
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 1000 \
    --benchmark_type ml_only

# Run only GTO Halo benchmarking
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 1000 \
    --benchmark_type gto_halo_only \
    --enable_physical_validation
```

## Available Checkpoints

The framework is configured to use the new Training Runs directory structure:

- **Training Runs**: `../Reflected-Diffusion/Training Runs/YYYY.MM.DD_HHMMSS/`
- **Checkpoints**: Located in `checkpoints/` subdirectory of each training run
- **Logs**: Located in `logs/` subdirectory of each training run

The framework automatically detects the most recent training run, or you can specify a specific run directory.

## Features

### ML Statistics (`ml_statistics.py`)
- **Sample Quality Metrics**: FID, IS, Precision, Recall
- **Distribution Analysis**: KL divergence, Wasserstein distance
- **Statistical Tests**: Kolmogorov-Smirnov, Anderson-Darling
- **Visualization**: Histograms, scatter plots, correlation matrices

### GTO Halo Benchmarking (`gto_halo_benchmarking.py`)
- **Physical Validation**: CR3BP feasibility checks
- **Trajectory Analysis**: Fuel consumption, time-of-flight
- **Constraint Satisfaction**: Boundary conditions, mass constraints
- **Optimization Metrics**: Local optimality, convergence analysis

## Configuration

### Model Configuration
```python
config = {
    'model_path': '../Reflected-Diffusion/Training Runs/2025.01.15_143022',  # Checkpoint directory
    'data_path': 'GTO_Halo_DM/data/training_data_boundary_100000.pkl',  # Reference data
    'num_samples': 1000,  # Number of samples to generate
    'batch_size': 100,  # Batch size for sampling
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### Physical Validation Configuration
```json
{
    "cr3bp_params": {
        "mu": 0.01215058560962404,
        "L1_distance": 0.146347522,
        "L2_distance": 0.167832734
    },
    "constraints": {
        "max_thrust": 0.1,
        "min_mass": 0.8,
        "max_time": 10.0
    }
}
```

## Output Structure

```
benchmark_results/
├── ml_statistics/
│   ├── samples.npy
│   ├── metrics.json
│   ├── plots/
│   └── summary.txt
├── gto_halo/
│   ├── samples.npy
│   ├── physical_metrics.json
│   ├── plots/
│   └── summary.txt
└── combined_report.txt
```

## Dependencies

```bash
pip install torch torchvision
pip install numpy scipy matplotlib seaborn
pip install scikit-learn
pip install pydylan  # For CR3BP validation
```

## Usage Examples

### Quick Test
```bash
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 10 \
    --test_mode
```

### Full Evaluation
```bash
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 1000 \
    --benchmark_type both \
    --enable_physical_validation \
    --save_samples \
    --save_plots \
    --verbose
```

### Custom Configuration
```bash
python run_benchmark.py \
    --model_path ../Reflected-Diffusion/Training\ Runs/2025.01.15_143022 \
    --data_path GTO_Halo_DM/data/training_data_boundary_100000.pkl \
    --num_samples 500 \
    --batch_size 50 \
    --sampling_method pc \
    --guidance_weight 0.1 \
    --benchmark_type both \
    --enable_physical_validation \
    --cr3bp_config custom_cr3bp_config.json
```

## Troubleshooting

1. **Checkpoint Not Found**: Ensure the model_path points to a directory with checkpoint files
2. **Physical Validation Errors**: Verify pydylan installation and CR3BP configuration
3. **Memory Issues**: Reduce batch_size or num_samples
4. **CUDA Errors**: Set device to 'cpu' if GPU memory is insufficient

## Notes

- The framework automatically uses the latest available checkpoint (checkpoint_3.pth)
- Physical validation requires the pydylan package for CR3BP simulations
- Results are saved in both JSON and human-readable formats
- The framework includes comprehensive error handling and logging 