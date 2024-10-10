#!/bin/bash

echo "Experiment started."
date

source venv/bin/activate

results_dir="$HOME/Results/N-player\ CC\ MAPPO"
python benchmarl/run.py algorithm=mappo task=hierarchial/chemical_production \
    experiment.loggers="[csv]" \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.save_folder="$results_dir" \
    experiment.max_n_frames=6000000 \
    experiment.evaluation_interval=600000 \
    task.safety_violation_penalty=1600 \
    seed=$RANDOM

echo "Experiment ended"
date
