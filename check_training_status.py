#!/usr/bin/env python
"""
Quick script to check training status
"""
import os
import glob
import json

# Check for training output directories
clothing_dir = "Clothing"
if os.path.exists(clothing_dir):
    print(f"[OK] Training output directory exists: {clothing_dir}")
    
    # Find all subdirectories (experiment folders)
    subdirs = [d for d in os.listdir(clothing_dir) if os.path.isdir(os.path.join(clothing_dir, d))]
    if subdirs:
        print(f"[OK] Found {len(subdirs)} experiment folder(s)")
        for subdir in subdirs:
            exp_path = os.path.join(clothing_dir, subdir)
            print(f"\n  Experiment: {subdir}")
            
            # Check for log files
            log_files = glob.glob(os.path.join(exp_path, "*.txt"))
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                print(f"    Latest log: {os.path.basename(latest_log)}")
                # Show last few lines
                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"    Last 3 lines:")
                            for line in lines[-3:]:
                                print(f"      {line.strip()}")
                except:
                    pass
            
            # Check for model files
            model_files = glob.glob(os.path.join(exp_path, "*.pth"))
            if model_files:
                print(f"    Model files: {len(model_files)}")
    else:
        print(f"[INFO] No experiment folders yet - training may be starting...")
else:
    print(f"[INFO] Training output directory not created yet - training may be starting...")

# Check for MM folder (summary logs)
mm_dir = os.path.join(clothing_dir, "MM")
if os.path.exists(mm_dir):
    print(f"\n[OK] Summary logs directory exists: {mm_dir}")

print("\n" + "="*60)
print("To monitor training in real-time, check the log files in:")
print(f"  {os.path.abspath(clothing_dir)}/")
print("="*60)

