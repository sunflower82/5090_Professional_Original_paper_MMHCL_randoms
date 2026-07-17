#!/usr/bin/env python
"""
Convert .inter file format to MMHCL JSON format
Converts MMRec format (.inter) to LATTICE/MMSSL format (JSON with 5-core)
"""

import json
import os
from collections import defaultdict

def convert_inter_to_json(inter_file, output_dir, dataset_name, core=5):
    """
    Convert .inter file to JSON format required by MMHCL.
    
    Args:
        inter_file: Path to .inter file
        output_dir: Output directory (e.g., 'data/Clothing')
        dataset_name: Dataset name
        core: Core filtering value (default: 5)
    """
    print(f"Reading {inter_file}...")
    
    # Read the .inter file
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)
    
    with open(inter_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline().strip().split('\t')
        print(f"Columns: {header}")
        
        # Find column indices
        user_idx = header.index('userID')
        item_idx = header.index('itemID')
        label_idx = header.index('x_label')
        
        line_count = 0
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Processed {line_count} lines...")
            
            parts = line.strip().split('\t')
            if len(parts) < max(user_idx, item_idx, label_idx) + 1:
                continue
            
            try:
                user_id = str(int(float(parts[user_idx])))
                item_id = int(float(parts[item_idx]))
                x_label = int(float(parts[label_idx]))
                
                if x_label == 0:
                    train_data[user_id].append(item_id)
                elif x_label == 1:
                    val_data[user_id].append(item_id)
                elif x_label == 2:
                    test_data[user_id].append(item_id)
            except (ValueError, IndexError):
                continue
    
    print(f"Total interactions processed: {line_count}")
    
    print(f"\nBefore core filtering:")
    print(f"  Train users: {len(train_data)}")
    print(f"  Val users: {len(val_data)}")
    print(f"  Test users: {len(test_data)}")
    
    # Apply 5-core filtering: keep only users/items with at least 'core' interactions
    if core > 0:
        print(f"\nApplying {core}-core filtering...")
        
        # Count item frequencies in training set
        item_counts = defaultdict(int)
        for items in train_data.values():
            for item in items:
                item_counts[item] += 1
        
        # Filter items with at least 'core' interactions
        valid_items = {item for item, count in item_counts.items() if count >= core}
        print(f"  Valid items (>= {core} interactions): {len(valid_items)}")
        
        # Filter users with at least 'core' interactions
        def filter_user_items(user_items):
            filtered = [item for item in user_items if item in valid_items]
            return filtered if len(filtered) >= core else []
        
        train_data = {uid: filter_user_items(items) 
                     for uid, items in train_data.items() 
                     if len(filter_user_items(items)) >= core}
        
        # Filter val and test data to only include items in valid_items and users in train_data
        val_data = {uid: [item for item in items if item in valid_items]
                   for uid, items in val_data.items() if uid in train_data}
        test_data = {uid: [item for item in items if item in valid_items]
                    for uid, items in test_data.items() if uid in train_data}
        
        print(f"\nAfter {core}-core filtering:")
        print(f"  Train users: {len(train_data)}")
        print(f"  Val users: {len(val_data)}")
        print(f"  Test users: {len(test_data)}")
    
    # Create output directory
    core_dir = os.path.join(output_dir, f'{core}-core')
    os.makedirs(core_dir, exist_ok=True)
    
    # Save JSON files
    train_file = os.path.join(core_dir, 'train.json')
    val_file = os.path.join(core_dir, 'val.json')
    test_file = os.path.join(core_dir, 'test.json')
    
    print(f"\nSaving JSON files to {core_dir}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    print(f"  [OK] Saved {train_file}")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f)
    print(f"  [OK] Saved {val_file}")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    print(f"  [OK] Saved {test_file}")
    
    print(f"\n[OK] Conversion complete!")
    return core_dir

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert .inter file to MMHCL JSON format')
    parser.add_argument('--inter_file', type=str, required=True,
                        help='Path to .inter file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (e.g., data/Clothing)')
    parser.add_argument('--dataset', type=str, default='Clothing',
                        help='Dataset name')
    parser.add_argument('--core', type=int, default=5,
                        help='Core filtering value (default: 5)')
    
    args = parser.parse_args()
    
    convert_inter_to_json(args.inter_file, args.output_dir, args.dataset, args.core)

