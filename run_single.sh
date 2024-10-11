#!/bin/bash

echo "Experiment started."
date

source venv/bin/activate

# Duplicate magic string.
# BenchMARL requires the results dir to be quoted and space-escaped,
# while mkdir -p requires it to be either quoted or space-escaped, but not both.
results_dir="$HOME/Results/N-player\ CP\ MAPPO"
mkdir -p "$HOME/Results/N-player CP MAPPO"

python benchmarl/run.py algorithm=mappo task=hierarchial/chemical_production \
    experiment.loggers="[csv]" \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.save_folder="$results_dir" \
    experiment.max_n_frames=6000000 \
    experiment.evaluation_interval=600000 \
    task.safety_violation_penalty=25600 \
    seed=$RANDOM

echo "Experiment ended"
date
