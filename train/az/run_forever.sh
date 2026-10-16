#!/bin/bash
# keeps the trainer running: restarts it (it resumes from its checkpoints) unless it stopped because of the STOP file
cd "$(dirname "$0")/../.."
while true; do
  python3 train/az/train.py >> train/az/out/stdout.txt 2>&1
  [ -f train/az/out/STOP ] && { echo "$(date) stopped by STOP file" >> train/az/out/stdout.txt; break; }
  echo "$(date) trainer exited, restarting in 30s" >> train/az/out/stdout.txt; sleep 30
done
