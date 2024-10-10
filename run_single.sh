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
    experiment.max_n_frames=200000 \
    experiment.evaluation_interval=20000 \
    experiment.on_policy_collected_frames_per_batch=300 \
    experiment.on_policy_n_envs_per_worker=3 \
    task.safety_violation_penalty=1600 \
    seed=$RANDOM

echo "Experiment ended"
date
