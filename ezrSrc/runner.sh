#!/bin/bash
for file in ../Data/*/*.csv; do
    basename=$(basename "$file")
    echo "Running $basename..."
    python corr.py "$file" > "causal_results/$basename"
done
