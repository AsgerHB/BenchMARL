#!/bin/bash
ARGS="--out=outputs/out.txt -e outputs.error.txt --partition=dhabi -n1 --mem=16G --job-name $job_name"

sbatch $ARGS ./run_single.sh