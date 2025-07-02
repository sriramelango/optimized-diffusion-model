# Optimized Diffusion Model for GTO Halo Trajectories

This repository contains an implementation of a Reflected Diffusion Model (RDM) adapted for generating GTO (Geosynchronous Transfer Orbit) Halo trajectory data. The project uses a 2D NCSN++ model with classifier-free guidance to learn and generate trajectory patterns.

## 🚀 Project Overview

The project adapts the Reflected Diffusion Model framework to train on GTO Halo trajectory data, which consists of `.pkl` files with `[N, 67]` arrays where the first value is a classifier and the remaining 66 values are trajectory features. The data is reshaped to `[1, 8, 9]` 2D images for processing.

## 📁 Repository Structure

```
Optimized Diffusion Model/
├── GTO_Halo_DM/                    # Original GTO Halo data and scripts
│   ├── data/                       # Training data
│   ├── data_generation_scripts/    # Data preprocessing scripts
│   └── DM_scripts/                 # Original diffusion model scripts
├── Reflected-Diffusion/            # Main RDM implementation
│   ├── configs/                    # Configuration files
│   │   ├── data/                   # Dataset configurations
│   │   ├── model/                  # Model architectures
│   │   └── train.yaml              # Training configuration
│   ├── models/                     # Model implementations
│   │   ├── ncsnpp.py              # NCSN++ model (modified for 2D)
│   │   ├── layerspp.py            # Layer implementations
│   │   └── utils.py               # Model utilities
│   ├── cube.py                    # Reflection functions
│   ├── losses.py                  # Training loss functions
│   ├── sampling.py                # Sampling algorithms
│   ├── sde_lib.py                 # SDE implementations
│   └── run_train.py               # Training script
├── runs/                          # Training outputs (gitignored)
└── README.md                      # This file
```

## 🔧 Key Features

### 1. **Reflected Diffusion Model**
- Implements domain-constrained diffusion with reflection
- Uses periodic reflection to keep samples in [0,1]^D domain
- Proper heat kernel score computation for reflected processes

### 2. **2D NCSN++ Architecture**
- Modified U-Net architecture for 2D image processing
- Classifier-free guidance for conditional generation
- Proper skip connection handling for 2D convolutions

### 3. **GTO Halo Data Processing**
- Reshapes trajectory data from [N, 67] to [1, 8, 9] images
- Extracts classifier labels for conditional training
- Maintains data integrity through preprocessing pipeline

### 4. **Training Features**
- Variance Exploding SDE (VESDE) with reflection
- Adam optimizer with learning rate scheduling
- EMA (Exponential Moving Average) for model stability
- Comprehensive logging and checkpointing

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sriramelango/optimized-diffusion-model.git
   cd optimized-diffusion-model
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install matplotlib pandas numpy
   pip install hydra-core
   ```

3. **Prepare data:**
   - Place your GTO Halo data in `GTO_Halo_DM/data/`
   - Ensure data is in the expected format: `[N, 67]` arrays

## 🚀 Usage

### Training

```bash
python Reflected-Diffusion/run_train.py \
    data=gto_halo \
    model=ncsnpp \
    --config-dir=Reflected-Diffusion/configs \
    --config-name=train
```

### Configuration

Key training parameters in `Reflected-Diffusion/configs/train.yaml`:
- `batch_size: 128`
- `n_iters: 1300001`
- `snapshot_freq: 50000` (checkpoint every 50k steps)
- `snapshot_freq_for_preemption: 10000` (preemption checkpoint every 10k steps)

### Model Architecture

The NCSN++ model is configured in `Reflected-Diffusion/configs/model/ncsnpp.yaml`:
- `image_size: 8, image_width: 9` (2D image dimensions)
- `ch_mult: [1, 2, 2]` (channel multipliers)
- `num_res_blocks: 2` (residual blocks per resolution)
- `conditional: true` (classifier-free guidance)

## 📊 Training Progress

The model logs training and evaluation loss every 50 and 100 steps respectively. Training progress can be monitored through:

- **Log files**: `runs/GTOHaloImage/YYYY.MM.DD/HHMMSS/logs`
- **Checkpoints**: Saved every 10k/50k steps
- **Samples**: Generated every 50k steps

## 🔬 Technical Details

### Reflected Diffusion Implementation

The model uses a **post-processing reflection approach** rather than true reflected Brownian motion:

1. **Standard VESDE**: `dx = 0*dt + σ(t)*dW`
2. **Post-processing**: `x = reflect(x)` after each step
3. **Heat kernel score**: Computed using reflected heat kernel
4. **Domain constraint**: Samples kept in [0,1]^D via periodic reflection

### Key Components

- **`cube.reflect()`**: Periodic reflection function
- **`cube.score_hk()`**: Reflected heat kernel score
- **`ReflectedEulerMaruyamaPredictor`**: Reflected sampling predictor
- **`ReflectedLangevinCorrector`**: Reflected sampling corrector

## 📈 Results

The model shows promising training progress:
- **Loss reduction**: From ~17.7 to ~15.6 in early training
- **Stable convergence**: No divergence or explosion
- **Domain compliance**: All samples remain in [0,1]^D

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the Reflected Diffusion Model framework
- Adapted for GTO Halo trajectory generation
- Uses PyTorch and Hydra for training infrastructure

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This repository contains both the original GTO Halo diffusion model scripts and the new Reflected Diffusion Model implementation. The RDM version is the primary focus and represents the optimized approach for trajectory generation. 