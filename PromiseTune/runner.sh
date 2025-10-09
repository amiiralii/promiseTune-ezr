#!/bin/bash
for file in Data/*.csv; do
    basename=$(basename "$file")
    echo "Running $basename..."
    python PromiseTune.py "$basename" > "results/$basename"
done
