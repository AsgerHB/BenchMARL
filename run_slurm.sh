#!/bin/bash

base_args="--partition=turing -n1 --mem=16G --job-name benchmarl"

repetitions=10
for ((r=1; r<=$repetitions; r++)); do
    sbatch $base_args "--out=outputs/out$r.txt -e outputs/error$r.txt" ./run_single.sh
done
