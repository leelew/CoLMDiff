# CoLMDiff

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) with support for both DDPM and DDIM sampling methods.

## Features

- **U-Net Denoiser**: A configurable U-Net architecture with time embeddings and optional self-attention layers
- **Flexible Noise Schedules**: Support for linear, cosine, quadratic, and warmup noise schedules
- **Multiple Sampling Methods**: 
  - DDPM (Denoising Diffusion Probabilistic Model) sampling
  - DDIM (Denoising Diffusion Implicit Model) sampling for faster generation
- **Conditional Generation**: Support for both unconditional and conditional diffusion models
- **Multi-GPU Training**: Seamless multi-GPU support via DataParallel
- **Configurable Training**: YAML-based configuration for easy experimentation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- torchinfo
- tqdm
- PyYAML
- matplotlib (for tests)

Install dependencies:

```bash
pip install torch numpy torchinfo tqdm pyyaml matplotlib
```

## Project Structure

```
CoLMDiff/
├── src/
│   ├── base.py         # Base model class with device utilities
│   ├── config.py       # Configuration parser
│   ├── config.yaml     # Default configuration file
│   ├── data.py         # Dataset class for training
│   ├── denoiser.py     # U-Net denoiser architecture
│   ├── diffusion.py    # Gaussian diffusion process implementation
│   ├── eval.py         # Evaluation script
│   ├── model.py        # DDPM model wrapper
│   ├── train.py        # Training script
│   └── trainer.py      # Trainer class for training and sampling
└── test/
    └── lorenz96.py     # Lorenz-96 EnKF example
```

## Configuration

The model is configured via a YAML file. Key configuration options include:

### Model Configuration

```yaml
model:
  height: 28              # Input image height (assumes square images)
  in_channel: 1           # Number of input channels
  out_channel: 1          # Number of output channels
  time_embed: 64          # Time embedding dimension
  groups: 16              # Groups for GroupNorm
  dropout: 0.1            # Dropout rate
  mults: [1, 2]           # Channel multipliers for each resolution
  blocks: 1               # Number of ResNet blocks per resolution
  with_attn_height: [14]  # Resolutions at which to apply attention

  timesteps: 1000         # Number of diffusion steps
  objective: 'pred_noise' # Training objective ('pred_noise' or 'pred_x0')
  conditional: false      # Whether to use conditional generation
```

### Training Configuration

```yaml
train:
  init_method: 'kaiming'  # Weight initialization method
  noise_schedule:
    linear_start: 0.0001
    linear_end: 0.02
    schedule: 'linear'    # Options: 'linear', 'cosine', 'quad', 'warmup10', 'warmup50'
  
  optimizer: 'Adam'       # Options: 'Adam', 'AdamW'
  lr: 0.0001
  lr_schedule: 'MultiStepLR'
  grad_clip: 1.0
  epochs: 5
  batch_size: 16
```

### Sampling Configuration

```yaml
sample:
  solver: 'ddpm'          # Options: 'ddpm', 'ddim'
  ddim:
    ddim_sample_steps: 50
    ddim_eta: 0.0         # 0 for deterministic, 1 for stochastic
  method: 'p(x)'          # Options: 'p(x)', 'p(x|y):sdedit', 'p(x|y):cond'
```

## Usage

### Training

1. Prepare your training data as NumPy arrays:
   - `train_x_start.npy`: Training samples
   - `val_x_start.npy`: Validation samples
   - (Optional) `train_x_condition.npy` and `val_x_condition.npy` for conditional models

2. Run training:

```bash
cd src
python train.py --path_config config.yaml --gpu_ids 0,1 --phase train --path_data /path/to/data/
```

### Evaluation / Sampling

Generate samples from a trained model:

```bash
cd src
python eval.py --path_config config.yaml --gpu_ids 0,1 --phase eval
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--path_config` | Path to configuration YAML file | Required |
| `--gpu_ids` | Comma-separated GPU IDs | `1,2,3` |
| `--phase` | Mode: `train` or `eval` | `train` |
| `--path_data` | Path to dataset directory | `` |

## Architecture

### U-Net Denoiser

The denoiser uses a U-Net architecture with:
- Sinusoidal time embeddings
- Residual blocks with Group Normalization
- Optional self-attention at specified resolutions
- Skip connections between encoder and decoder

### Diffusion Process

The implementation supports:
- **Forward process**: `q(x_t | x_0)` - Adding noise according to a schedule
- **Reverse process**: `p(x_{t-1} | x_t)` - Denoising using the trained model
- **Training objectives**: Predict noise (`pred_noise`) or clean image (`pred_x0`)

## License

This project is provided as-is for research purposes.
