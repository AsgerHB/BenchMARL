#!/bin/bash

ARGS="--out=outputs/out.txt -e outputs/error.txt --partition=turing -n1 --mem=16G --job-name benchmarl"

repetitions=10
for ((r=1; r<=$repetitions; r++)); do
    sbatch $ARGS ./run_single.sh
done
