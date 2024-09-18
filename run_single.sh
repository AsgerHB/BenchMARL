#!/bin/bash

echo "Experiment started."
date

source venv/bin/activate

results_dir="$HOME/Results/N-player\ CC\ MAPPO"
python benchmarl/run.py algorithm=mappo task=hierarchial/cruise_control \
    experiment.loggers="[csv]" \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.save_folder="$results_dir" \
    task.safety_violation_penalty=1600 \
    seed=$RANDOM

echo "Experiment ended"
date
