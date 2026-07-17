# Quick Start - Automatic MMHCL Training

## Fast Setup (3 Steps)

### 1. Install Dependencies

```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 scipy==1.16.3 scikit-learn==1.6.1 tqdm
```

### 2. Verify Your Dataset

Ensure your dataset is in `MMHCL/data/{DatasetName}/` with:
- `5-core/train.json`, `val.json`, `test.json`
- `image_feat.npy`, `text_feat.npy`

### 3. Run Training

**Windows:**
```bash
train.bat
```

**Linux/Mac:**
```bash
chmod +x train.sh
./train.sh
```

**Or directly:**
```bash
python auto_train.py --dataset Clothing
```

## That's It! 🚀

The script will:
- ✅ Verify all package versions match requirements
- ✅ Check CUDA availability
- ✅ Verify dataset files
- ✅ Start training automatically

## Customize Training

```bash
python auto_train.py --dataset Clothing --epoch 500 --batch_size 1024 --lr 0.0005
```

See [AUTO_TRAIN_README.md](AUTO_TRAIN_README.md) for full documentation.

