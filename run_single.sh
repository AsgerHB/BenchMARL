#!/bin/bash

echo "Experiment started."
date

source venv/bin/activate

results_dir="$HOMEDIR/Results/N-player CC MAPPO"
python benchmarl/run.py algorithm=mappo task=hierarchial/cruise_control \
    experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.max_n_frames=20000 \
    experiment.evaluation_interval=5000 \
    experiment.checkpoint_interval=5000 \
    experiment.loggers="[csv]" \
    experiment.evaluation_episodes=400 \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.save_folder=$results_dir

echo "Experiment ended"
date
