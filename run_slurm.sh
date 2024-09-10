#!/bin/bash

ARGS="--out=outputs/out.txt -e outputs/error.txt --partition=turing -n1 --mem=16G --job-name benchmarl"

sbatch $ARGS ./run_single.sh
