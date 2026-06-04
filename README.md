# RAAP: Reference-Aligned Accuracy Prediction

Reference-Aligned Accuracy Prediction for domain shift scenarios.

## Project Structure

```
RAAP_org/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data_setup/                    # Dataset download scripts
│   ├── setup_CIFAR10.sh
│   ├── setup_CIFAR100.sh
│   ├── setup_MNIST.sh
│   ├── setup_ImageNet.sh
│   ├── setup_TinyImagenet.sh
│   ├── setup_BREEDs.sh
│   └── ImageNet/
│       └── ImagenetV2_reorg.py
├── data_loader.py                 # Data loading utilities for different datasets
├── data_sets.py                   # Custom dataset classes
├── model_load.py                  # Model architecture loading
├── model_train.py                 # Model training script
├── Calibrate.py                   # Temperature scaling calibration
├── baseline.py                    # Baseline estimation methods
├── reference_aligned_pred.py      # FRAP (Fast Reference-Aligned Prediction) method
├── estimate.py                    # Main estimation pipeline
├── Result_Analysis.py             # Results analysis
└── utils.py                       # Utility functions
```

## File Descriptions

- **data_loader.py**: Handles data loading for multiple datasets (CIFAR10, CIFAR100, MNIST, ImageNet, TinyImageNet, BREEDS, Camelyon17, Fmow, Rxrx1, Amazon, CivilComments, DomainNet)
- **data_sets.py**: Custom PyTorch dataset implementations
- **model_load.py**: Model architecture factory (ResNet, DenseNet, VGG, DistilBERT, SimpleConv)
- **model_train.py**: Main training script with multi-GPU support
- **Calibrate.py**: Temperature scaling for confidence calibration
- **baseline.py**: Various baseline estimation methods (IM, histogram-based, EMD, etc.)
- **reference_aligned_pred.py**: FRAP method implementation using reference models
- **estimate.py**: Main estimation pipeline that combines models and methods
- **Result_Analysis.py**: Statistical analysis and visualization of results
- **utils.py**: Common utility functions (optimizers, schedulers, metrics, etc.)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

Use the provided scripts in `data_setup/` to download and organize datasets:

```bash
cd data_setup

# For CIFAR-10
./setup_CIFAR10.sh <data_dir>

# For CIFAR-100
./setup_CIFAR100.sh <data_dir>

# For MNIST
./setup_MNIST.sh <data_dir>

# For ImageNet (requires manual download)
./setup_Imagenet.sh <data_dir>

# For TinyImageNet
./setup_TinyImagenet.sh <data_dir>

# For ImageNet-based BREEDS
./setup_BREEDs.sh <data_dir>
```

Replace `<data_dir>` with your desired data directory path.

## Quick Start

### Training a Model

```bash
python model_train.py \
    --dataset CIFAR10 \
    --model ResNet18 \
    --data_path /path/to/data \
    --batch_size 64 \
    --train_epoch 20 \
    --pretrained
```

**Arguments:**
- `--dataset`: Dataset name (CIFAR10, CIFAR100, MNIST, ImageNet, TinyImageNet, etc.)
- `--model`: Model architecture (ResNet18, ResNet50, DenseNet121, VGG11, distilbert-base-uncased)
- `--data_path`: Path to dataset directory
- `--batch_size`: Batch size for training
- `--lr`: Learning rate (default: 0.001)
- `--train_epoch`: Number of training epochs (default: 20)
- `--pretrained`: Use pretrained weights
- `--model_seed`: Random seed for model initialization

### Estimating Performance on Distribution Shifts

```bash
python estimate.py \
    --dataset CIFAR10 \
    --model ResNet18 \
    --metric IM \
    --data_path /path/to/data \
    --ckpt_epoch 20 \
    --pretrained
```

**Arguments:**
- `--dataset`: Dataset name
- `--model`: Model architecture
- `--metric`: Estimation metric (IM, HistDensity, EMD, etc.)
- `--data_path`: Path to dataset directory
- `--ckpt_epoch`: Checkpoint epoch to load
- `--batch_size`: Batch size for evaluation (default: 128)
- `--pretrained`: Use pretrained weights
- `--model_seed`: Random seed

### Analyzing Results

```bash
python Result_Analysis.py \
    --dataset CIFAR10 \
    --metric IM \
    --pretrained
```

## Supported Datasets

- **Vision**: CIFAR-10, CIFAR-100, MNIST, ImageNet, TinyImageNet, DomainNet
- **BREEDS**: living17, nonliving26, entity13, entity30
- **Wilds**: Camelyon17, Fmow, Rxrx1, Amazon, CivilComments

## Notes

- Modify `CUDA_VISIBLE_DEVICES` in scripts based on your hardware configuration
- Update `data_path` arguments to point to your dataset location
- Use `--pretrained` flag for faster training on smaller datasets
- Results are saved in checkpoints and analysis directories

## Citation

[Add your citation information here]

## Contact

[Add contact information here]
