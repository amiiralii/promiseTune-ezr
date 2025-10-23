#!/usr/bin/env python3
"""
Test script to demonstrate the causality-based tree vs standard tree
"""

from causal_tools import CausalTree, h, hcond
from ezr import *
import sys

def test_causal_vs_standard():
    """Compare causality-based tree with standard tree"""
    
    # Load sample data
    filename = "../Data/moot/auto93.csv"
    data = Data(csv(filename))
    
    print("=== CAUSALITY-BASED TREE vs STANDARD TREE COMPARISON ===")
    print(f"Dataset: {filename}")
    print(f"Rows: {len(data.rows)}")
    print(f"Columns: {len(data.cols.x)}")
    print()
    
    # Split data
    half = len(data.rows) // 2
    data.rows = shuffle(data.rows)
    tr, holdout = data.rows[:half], data.rows[half:]
    train = likely(clone(data, tr))
    
    # Build standard tree
    print("Building standard tree...")
    standard_tree = Tree(clone(data, train))
    
    # Build causality-based tree
    print("Building causality-based tree...")
    causal_tree = CausalTree(clone(data, train))
    
    # Compare tree structures
    print("\n=== TREE COMPARISON ===")
    print("Standard Tree:")
    treeShow(data, standard_tree, lambda v: int(100*(1 - (v - adds(disty(data,row) for row in data.rows).lo)/(adds(disty(data,row) for row in data.rows).mu - adds(disty(data,row) for row in data.rows).lo))))
    
    print("\nCausality-based Tree:")
    treeShow(data, causal_tree, lambda v: int(100*(1 - (v - adds(disty(data,row) for row in data.rows).lo)/(adds(disty(data,row) for row in data.rows).mu - adds(disty(data,row) for row in data.rows).lo))))
    
    # Test predictions on holdout set
    print("\n=== PREDICTION COMPARISON ===")
    standard_predictions = [treeLeaf(standard_tree, row).mu for row in holdout]
    causal_predictions = [treeLeaf(causal_tree, row).mu for row in holdout]
    actual_values = [disty(data, row) for row in holdout]
    
    # Calculate errors
    standard_errors = [abs(pred - actual) for pred, actual in zip(standard_predictions, actual_values)]
    causal_errors = [abs(pred - actual) for pred, actual in zip(causal_predictions, actual_values)]
    
    print(f"Standard Tree - Mean Error: {sum(standard_errors)/len(standard_errors):.3f}")
    print(f"Causality Tree - Mean Error: {sum(causal_errors)/len(causal_errors):.3f}")
    
    # Show causality analysis for each column
    print("\n=== CAUSALITY ANALYSIS ===")
    d2hs = adds(disty(data,row) for row in train)
    d2hs.rows = [disty(data,row) for row in train]
    
    for col in data.cols.x:
        x = [r[col.at] for r in train]
        y = d2hs.rows
        
        # Calculate causality measures
        h_x = h(x)
        h_y = h(y)
        h_x_given_y = hcond(x, y)
        h_y_given_x = hcond(y, x)
        
        causality_ratio_x = h_x_given_y / h_x if h_x > 0 else 1
        causality_ratio_y = h_y_given_x / h_y if h_y > 0 else 1
        
        print(f"{col.txt}:")
        print(f"  H(X)={h_x:.3f}, H(Y)={h_y:.3f}")
        print(f"  H(X|Y)={h_x_given_y:.3f}, H(Y|X)={h_y_given_x:.3f}")
        print(f"  Causality ratios: X->Y={causality_ratio_x:.3f}, Y->X={causality_ratio_y:.3f}")
        print(f"  Better causality: {'X->Y' if causality_ratio_x < causality_ratio_y else 'Y->X'}")
        print()

if __name__ == "__main__":
    test_causal_vs_standard()
