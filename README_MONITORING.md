# Training Monitoring in Cursor IDE

## Quick Status Check

Run this command in the terminal to see current training status:

```bash
python quick_status.py
```

## Real-time Monitoring

### Option 1: Interactive Monitor (Recommended)

Run the interactive monitor that updates in real-time:

```bash
python monitor_training.py
```

This will:
- Display training progress as it happens
- Show metrics (Recall, Precision, NDCG, Hit Ratio) at @10 and @20
- Track best metrics
- Show test results when validation improves
- Update every 2 seconds

**Press Ctrl+C to stop monitoring**

### Option 2: Watch Log File Directly

In Cursor IDE, you can:
1. Open the log file: `Clothing/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_.txt`
2. The file will auto-refresh as training progresses
3. Full metrics appear every 5 epochs (since verbose=5)

### Option 3: PowerShell Tail Command

Watch the log file in real-time:

```powershell
Get-Content "Clothing\uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_\uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_.txt" -Wait -Tail 20
```

## Understanding the Log Format

### Epoch Logs (Every 5 epochs)
```
Epoch 5 [X.Xs + X.Xs]: train==[loss=mf_loss + emb_loss + contrastive_loss], 
recall=[@10, @20], precision=[@10, @20], hit=[@10, @20], ndcg=[@10, @20]
```

### Test Results (When validation improves)
```
Test_Recall@20: X.XXXX   Test_Precision@20: X.XXXX   Test_NDCG@20: X.XXXX
```

### Simple Epoch Logs (Other epochs)
```
Epoch X [X.Xs]: train==[loss=mf_loss + emb_loss + contrastive_loss]
```

## Current Training Status

- **Epoch 0**: Completed (took ~25 minutes)
- **Next Metrics**: Will appear at epoch 5
- **Total Epochs**: 1000 (or until early stopping)
- **Early Stopping**: If no improvement for 5 consecutive evaluations

## Tips

1. **Keep the monitor running** in a separate terminal tab
2. **Check periodically** with `python quick_status.py`
3. **Training is slow** - each epoch takes ~25 minutes, so be patient
4. **Metrics update every 5 epochs** - don't expect updates every epoch

