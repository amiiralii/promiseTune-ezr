#!/bin/bash
for file in Data/pt_data/*.csv; do
    basename=$(basename "$file")
    echo "Running $basename..."
    python experiment.py "$file" > "results/$basename"
done
