#!/usr/bin/env python
"""
Real-time MMHCL Training Monitor for Cursor IDE
Monitors training progress and displays metrics in real-time
"""

import os
import time
import re
from pathlib import Path
from datetime import datetime

class TrainingMonitor:
    def __init__(self, log_file_path):
        self.log_file = Path(log_file_path)
        self.last_position = 0
        self.epochs_seen = set()
        self.best_metrics = {
            'recall@10': 0.0,
            'recall@20': 0.0,
            'ndcg@10': 0.0,
            'ndcg@20': 0.0,
            'precision@10': 0.0,
            'precision@20': 0.0,
            'hit@10': 0.0,
            'hit@20': 0.0
        }
        self.test_results = {}
        
    def parse_epoch_line(self, line):
        """Parse training epoch log line"""
        # Pattern: Epoch X [Xs + Xs]: train==[loss=mf_loss + emb_loss + contrastive_loss], 
        # recall=[@10, @20], precision=[@10, @20], hit=[@10, @20], ndcg=[@10, @20]
        pattern = r'Epoch (\d+) \[([\d.]+)s \+ ([\d.]+)s\]: train==\[([\d.]+)=([\d.]+) \+ ([\d.]+) \+ ([\d.]+)\], ' \
                  r'recall=\[([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+)\], ' \
                  r'hit=\[([\d.]+), ([\d.]+)\], ndcg=\[([\d.]+), ([\d.]+)\]'
        
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            train_time = float(match.group(2))
            eval_time = float(match.group(3))
            total_loss = float(match.group(4))
            mf_loss = float(match.group(5))
            emb_loss = float(match.group(6))
            contrastive_loss = float(match.group(7))
            recall_10 = float(match.group(8))
            recall_20 = float(match.group(9))
            precision_10 = float(match.group(10))
            precision_20 = float(match.group(11))
            hit_10 = float(match.group(12))
            hit_20 = float(match.group(13))
            ndcg_10 = float(match.group(14))
            ndcg_20 = float(match.group(15))
            
            return {
                'epoch': epoch,
                'train_time': train_time,
                'eval_time': eval_time,
                'loss': {
                    'total': total_loss,
                    'mf': mf_loss,
                    'emb': emb_loss,
                    'contrastive': contrastive_loss
                },
                'recall': {'@10': recall_10, '@20': recall_20},
                'precision': {'@10': precision_10, '@20': precision_20},
                'hit': {'@10': hit_10, '@20': hit_20},
                'ndcg': {'@10': ndcg_10, '@20': ndcg_20}
            }
        return None
    
    def parse_test_line(self, line):
        """Parse test results line"""
        # Pattern: Test_Recall@20: X.XXXX   Test_Precision@20: X.XXXX   Test_NDCG@20: X.XXXX
        pattern = r'Test_Recall@(\d+): ([\d.]+)\s+Test_Precision@(\d+): ([\d.]+)\s+Test_NDCG@(\d+): ([\d.]+)'
        match = re.search(pattern, line)
        if match:
            k = int(match.group(1))
            recall = float(match.group(2))
            precision = float(match.group(4))
            ndcg = float(match.group(6))
            return {'k': k, 'recall': recall, 'precision': precision, 'ndcg': ndcg}
        return None
    
    def display_header(self):
        """Display monitoring header"""
        print("=" * 100)
        print("MMHCL Training Monitor - Real-time Progress")
        print("=" * 100)
        print(f"Log File: {self.log_file}")
        print(f"Status: {'Monitoring...' if self.log_file.exists() else 'Waiting for log file...'}")
        print("=" * 100)
        print()
    
    def display_epoch(self, epoch_data):
        """Display formatted epoch information"""
        epoch = epoch_data['epoch']
        self.epochs_seen.add(epoch)
        
        # Update best metrics
        if epoch_data['recall']['@20'] > self.best_metrics['recall@20']:
            self.best_metrics['recall@20'] = epoch_data['recall']['@20']
        if epoch_data['ndcg']['@20'] > self.best_metrics['ndcg@20']:
            self.best_metrics['ndcg@20'] = epoch_data['ndcg']['@20']
        if epoch_data['precision']['@20'] > self.best_metrics['precision@20']:
            self.best_metrics['precision@20'] = epoch_data['precision']['@20']
        
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch} | Train: {epoch_data['train_time']:.1f}s | Eval: {epoch_data['eval_time']:.1f}s")
        print(f"{'='*100}")
        
        # Loss breakdown
        print(f"Loss: {epoch_data['loss']['total']:.5f} = "
              f"MF: {epoch_data['loss']['mf']:.5f} + "
              f"Emb: {epoch_data['loss']['emb']:.5f} + "
              f"Contrastive: {epoch_data['loss']['contrastive']:.5f}")
        
        # Metrics table
        print(f"\n{'Metric':<15} {'@10':<12} {'@20':<12} {'Best@20':<12}")
        print("-" * 60)
        print(f"{'Recall':<15} {epoch_data['recall']['@10']:<12.6f} {epoch_data['recall']['@20']:<12.6f} {self.best_metrics['recall@20']:<12.6f}")
        print(f"{'Precision':<15} {epoch_data['precision']['@10']:<12.6f} {epoch_data['precision']['@20']:<12.6f} {self.best_metrics['precision@20']:<12.6f}")
        print(f"{'Hit Ratio':<15} {epoch_data['hit']['@10']:<12.6f} {epoch_data['hit']['@20']:<12.6f} {'-':<12}")
        print(f"{'NDCG':<15} {epoch_data['ndcg']['@10']:<12.6f} {epoch_data['ndcg']['@20']:<12.6f} {self.best_metrics['ndcg@20']:<12.6f}")
        
        print(f"\nTotal Epochs Completed: {len(self.epochs_seen)}")
        print(f"Epochs Seen: {sorted(list(self.epochs_seen))}")
    
    def display_test_results(self, test_data):
        """Display test results"""
        k = test_data['k']
        self.test_results[k] = test_data
        
        print(f"\n{'='*100}")
        print(f"TEST RESULTS @{k} (Best Model So Far)")
        print(f"{'='*100}")
        print(f"Recall@{k}:    {test_data['recall']:.8f}")
        print(f"Precision@{k}: {test_data['precision']:.8f}")
        print(f"NDCG@{k}:      {test_data['ndcg']:.8f}")
        print(f"{'='*100}")
    
    def display_summary(self):
        """Display training summary"""
        if self.test_results:
            print(f"\n{'='*100}")
            print("FINAL TEST RESULTS SUMMARY")
            print(f"{'='*100}")
            for k in sorted(self.test_results.keys()):
                data = self.test_results[k]
                print(f"\n@{k}:")
                print(f"  Recall:    {data['recall']:.8f}")
                print(f"  Precision: {data['precision']:.8f}")
                print(f"  NDCG:      {data['ndcg']:.8f}")
    
    def monitor(self, refresh_interval=2):
        """Monitor log file for updates"""
        self.display_header()
        
        if not self.log_file.exists():
            print(f"Waiting for log file to be created: {self.log_file}")
            print("Press Ctrl+C to exit")
            while not self.log_file.exists():
                time.sleep(refresh_interval)
        
        print(f"Monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                if self.log_file.exists():
                    with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        # Seek to last known position
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Check for epoch data
                            epoch_data = self.parse_epoch_line(line)
                            if epoch_data:
                                self.display_epoch(epoch_data)
                                continue
                            
                            # Check for test results
                            test_data = self.parse_test_line(line)
                            if test_data:
                                self.display_test_results(test_data)
                                continue
                            
                            # Check for early stopping
                            if 'Early stopping steps' in line or 'Early stop' in line:
                                print(f"\n{'!'*50}")
                                print(line)
                                print(f"{'!'*50}")
                            
                            # Check for other important messages
                            if 'ERROR' in line or 'error' in line.lower():
                                print(f"\n[ERROR] {line}")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*100)
            print("Monitoring stopped by user")
            self.display_summary()
            print("="*100)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor MMHCL training progress')
    parser.add_argument('--log_file', type=str, 
                       default='Clothing/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_/uu_ii=3_2_0.03_0.07_topk=5_t=0.6_regs=0.001_dim=64_.txt',
                       help='Path to training log file')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Refresh interval in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Resolve log file path relative to MMHCL directory
    script_dir = Path(__file__).parent
    log_file = script_dir / args.log_file
    
    monitor = TrainingMonitor(log_file)
    monitor.monitor(refresh_interval=args.interval)

if __name__ == '__main__':
    main()

