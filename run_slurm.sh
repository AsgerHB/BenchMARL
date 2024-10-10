#!/bin/bash

base_args="--partition=turing -n1 --mem=64G --job-name benchmarl"

repetitions=10
for ((r=1; r<=$repetitions; r++)); do
    sbatch $base_args "--out=outputs/out$r.txt" "--error=outputs/error$r.txt " ./run_single.sh
done
