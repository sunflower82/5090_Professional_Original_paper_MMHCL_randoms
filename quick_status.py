#!/usr/bin/env python
"""
Quick training status checker - shows current progress
"""
import os
import re
from pathlib import Path
from datetime import datetime

def get_training_status():
    log_file = Path("Clothing/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_.txt")
    
    if not log_file.exists():
        return "Log file not found. Training may not have started yet."
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        return "Training initializing..."
    
    # Get last few lines
    last_lines = lines[-10:]
    
    # Find latest epoch
    latest_epoch = None
    latest_metrics = None
    test_results = []
    
    for line in reversed(lines):
        # Check for epoch with metrics
        epoch_match = re.search(r'Epoch (\d+) \[([\d.]+)s \+ ([\d.]+)s\]: train==\[([\d.]+)=([\d.]+) \+ ([\d.]+) \+ ([\d.]+)\], recall=\[([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+)\], hit=\[([\d.]+), ([\d.]+)\], ndcg=\[([\d.]+), ([\d.]+)\]', line)
        if epoch_match:
            latest_epoch = int(epoch_match.group(1))
            latest_metrics = {
                'epoch': latest_epoch,
                'train_time': float(epoch_match.group(2)),
                'eval_time': float(epoch_match.group(3)),
                'total_loss': float(epoch_match.group(4)),
                'recall_10': float(epoch_match.group(8)),
                'recall_20': float(epoch_match.group(9)),
                'precision_10': float(epoch_match.group(10)),
                'precision_20': float(epoch_match.group(11)),
                'hit_10': float(epoch_match.group(12)),
                'hit_20': float(epoch_match.group(13)),
                'ndcg_10': float(epoch_match.group(14)),
                'ndcg_20': float(epoch_match.group(15))
            }
            break
        
        # Check for simple epoch (no metrics yet)
        simple_epoch = re.search(r'Epoch (\d+) \[([\d.]+)s\]: train==\[([\d.]+)=([\d.]+) \+ ([\d.]+) \+ ([\d.]+)\]', line)
        if simple_epoch and latest_epoch is None:
            latest_epoch = int(simple_epoch.group(1))
            latest_metrics = {
                'epoch': latest_epoch,
                'train_time': float(simple_epoch.group(2)),
                'total_loss': float(simple_epoch.group(3))
            }
        
        # Check for test results
        test_match = re.search(r'Test_Recall@(\d+): ([\d.]+)\s+Test_Precision@(\d+): ([\d.]+)\s+Test_NDCG@(\d+): ([\d.]+)', line)
        if test_match:
            test_results.append({
                'k': int(test_match.group(1)),
                'recall': float(test_match.group(2)),
                'precision': float(test_match.group(4)),
                'ndcg': float(test_match.group(6))
            })
    
    # Build status message
    status = []
    status.append("=" * 80)
    status.append("MMHCL Training Status")
    status.append("=" * 80)
    
    if latest_epoch is not None:
        status.append(f"Current Epoch: {latest_epoch}")
        if latest_metrics and 'recall_20' in latest_metrics:
            status.append(f"\nLatest Metrics (Epoch {latest_epoch}):")
            status.append(f"  Recall@10:    {latest_metrics['recall_10']:.6f}")
            status.append(f"  Recall@20:    {latest_metrics['recall_20']:.6f}")
            status.append(f"  Precision@10: {latest_metrics['precision_10']:.6f}")
            status.append(f"  Precision@20: {latest_metrics['precision_20']:.6f}")
            status.append(f"  NDCG@10:      {latest_metrics['ndcg_10']:.6f}")
            status.append(f"  NDCG@20:      {latest_metrics['ndcg_20']:.6f}")
            status.append(f"  Hit@10:       {latest_metrics['hit_10']:.6f}")
            status.append(f"  Hit@20:       {latest_metrics['hit_20']:.6f}")
            status.append(f"  Loss:          {latest_metrics['total_loss']:.5f}")
        else:
            status.append(f"  Training in progress... (Loss: {latest_metrics.get('total_loss', 'N/A')})")
            status.append(f"  Full metrics will appear at epoch 5, 10, 15, etc.")
    else:
        status.append("Training initializing or in early stages...")
    
    if test_results:
        status.append(f"\nBest Test Results:")
        for result in test_results:
            status.append(f"  @{result['k']}: Recall={result['recall']:.6f}, Precision={result['precision']:.6f}, NDCG={result['ndcg']:.6f}")
    
    status.append("=" * 80)
    status.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(status)

if __name__ == '__main__':
    print(get_training_status())

