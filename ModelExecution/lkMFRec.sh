#!/bin/bash

# List of subset sizes
#subsetSizes=(5000 10000 50000 100000 500000)
 
#list of models:
models=("ImpMF" "popular" "SVDBiased" "SVDfunk")

#number of recommendations
numRecs=10
for model in "${models[@]}"
do
    echo "Model: $model"
    # Run Python script with arguments
    python ../models/lkpyRecommendations.py $model $numRecs
done