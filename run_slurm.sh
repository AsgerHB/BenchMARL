#!/bin/bash

base_args="--out=/dev/null --error=/dev/null --partition=turing -n1 --mem=64G --job-name benchmarl"

repetitions=9
for ((r=1; r<=$repetitions; r++)); do
    sbatch $base_args ./run_single.sh
done
