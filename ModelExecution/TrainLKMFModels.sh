#!/bin/bash

# List of subset sizes
subsetSizes=(5000 10000 50000 100000 500000)

# Number of iterations and features
numIterations=1000
numFeatures=10

# Iterate over subset sizes
for subsetSize in "${subsetSizes[@]}"
do
    # Run Python script with arguments
    python ../models/lkpyMF.py $subsetSize $numIterations $numFeatures
done
