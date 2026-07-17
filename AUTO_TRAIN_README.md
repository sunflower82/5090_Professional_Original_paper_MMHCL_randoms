# Automatic MMHCL Training Guide

This guide explains how to use the automatic training script for the MMHCL model with specific package versions.

## Requirements

The automatic training script verifies the following package versions:

- **Python**: 3.12.12
- **PyTorch**: 2.2.0+cu118 (with CUDA 11.8 support)
- **NumPy**: 1.26.4
- **SciPy**: 1.16.3
- **scikit-learn**: 1.6.1
- **CUDA**: Available (for GPU training)

## Installation

### Option 1: Using requirements.txt

```bash
pip install -r requirements.txt
```

Note: For PyTorch with CUDA 11.8, you may need to install it separately:
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 scipy==1.16.3 scikit-learn==1.6.1 tqdm
```

### Option 2: Manual Installation

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy==1.26.4 scipy==1.16.3 scikit-learn==1.6.1 tqdm
```

## Dataset Setup

Ensure your dataset is properly structured. The expected structure is:

```
MMHCL/
├── data/
│   ├── Clothing/          # or Sports, Tiktok
│   │   ├── 5-core/
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   └── test.json
│   │   ├── image_feat.npy
│   │   └── text_feat.npy
│   └── ...
└── codes/
    └── ...
```

For TikTok dataset, also include `audio_feat.npy`.

## Usage

### Basic Usage

Run the automatic training script with default settings:

**Windows:**
```bash
train.bat
```

**Linux/Mac:**
```bash
python auto_train.py
```

### Advanced Usage

You can customize training parameters:

```bash
python auto_train.py \
    --dataset Clothing \
    --gpu_id 0 \
    --epoch 1000 \
    --batch_size 1024 \
    --lr 0.0001 \
    --embed_size 64 \
    --topk 5 \
    --User_layers 3 \
    --Item_layers 2 \
    --user_loss_ratio 0.03 \
    --item_loss_ratio 0.07 \
    --temperature 0.6
```

### Command Line Arguments

#### Dataset Options
- `--dataset`: Dataset name (choices: Clothing, Sports, Tiktok) [default: Clothing]
- `--gpu_id`: GPU ID to use [default: 0]

#### Training Hyperparameters
- `--epoch`: Number of training epochs [default: 1000]
- `--verbose`: Evaluation interval [default: 5]
- `--batch_size`: Batch size [default: 1024]
- `--lr`: Learning rate [default: 0.0001]
- `--regs`: Regularization [default: 1e-3]
- `--embed_size`: Embedding dimension [default: 64]
- `--topk`: K-NN sparsification [default: 5]
- `--core`: Core filtering [default: 5]
- `--User_layers`: User hypergraph layers [default: 3]
- `--Item_layers`: Item hypergraph layers [default: 2]
- `--user_loss_ratio`: User contrastive loss weight [default: 0.03]
- `--item_loss_ratio`: Item contrastive loss weight [default: 0.07]
- `--temperature`: InfoNCE temperature [default: 0.6]

#### Verification Options
- `--skip_verification`: Skip environment verification
- `--skip_dataset_check`: Skip dataset availability check

### Examples

**Train on Sports dataset:**
```bash
python auto_train.py --dataset Sports
```

**Train with custom hyperparameters:**
```bash
python auto_train.py --dataset Clothing --epoch 500 --batch_size 1024 --lr 0.0005
```

**Skip verification (if you're sure about the environment):**
```bash
python auto_train.py --skip_verification --skip_dataset_check
```

## What the Script Does

1. **Environment Verification**: Checks that all required packages are installed with correct versions
2. **CUDA Check**: Verifies CUDA availability and displays GPU information
3. **Dataset Check**: Verifies that all required dataset files are present
4. **Training**: Runs the MMHCL training with specified parameters
5. **Logging**: Training logs are saved in the dataset-specific directory

## Output

Training results and model weights are saved in:
```
MMHCL/{dataset}/{path_name}/
```

Where `path_name` is constructed from hyperparameters:
```
uu_ii={User_layers}_{Item_layers}_{user_loss_ratio}_{item_loss_ratio}_topk={topk}_t={temperature}_regs={regs}_dim={embed_size}_{ablation_target}
```

## Troubleshooting

### Version Mismatch

If you see version mismatch warnings:
1. Install the exact versions specified in `requirements.txt`
2. Restart your Python environment
3. Verify versions: `python auto_train.py` (without skipping verification)

### NumPy 2.x Issues

If you encounter NumPy 2.x compatibility issues:
```bash
pip install "numpy<2.0" --force-reinstall
```
Then restart your Python environment.

### Dataset Not Found

If the dataset check fails:
1. Verify the dataset structure matches the expected format
2. Check that files are in the correct location
3. Use `--skip_dataset_check` if you're certain the dataset is correct

### CUDA Not Available

If CUDA is not available:
- The script will warn you but continue with CPU training (very slow)
- Ensure you have CUDA 11.8 installed
- Verify GPU drivers are up to date

### Out of Memory

If you run out of GPU memory:
- Reduce batch size: `--batch_size 512` or `--batch_size 256`
- Reduce embedding size: `--embed_size 32`

## Notes

- The script automatically changes to the `codes` directory before running training
- Training logs are saved automatically
- Early stopping is enabled by default (patience: 5 epochs)
- Model weights are saved when validation performance improves

## Support

For more information about MMHCL, see the main [README.md](README.md).

