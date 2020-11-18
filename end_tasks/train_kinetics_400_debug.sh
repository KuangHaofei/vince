#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_kinetics_400"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --solver EndTaskKinetics400Solver \
  --lr-decay-type step \
  --lr-step-schedule 10 20 30 \
  --epochs 40 \
  --dataset Kinetics400Dataset \
  --data-path /home/ubuntu/drive2/kinetics400 \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 0 \
  --base-logdir ${BASE_LOG_LOCATION} \
  --backbone ResNet18 \
  --batch-size 16 \
  --num-frames 10 \
  --title ${TITLE} \
  --iterations-per-epoch 5000 \
  --image-log-frequency 5000 \
  --save-frequency 5000 \
  --long-save-frequency 25 \
  --num-workers 0 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --base-lr 0.01 \
  --no-save \
  --description kiana_1 \
  --end-task-classifier-num-classes 400 \
  --freeze-feature-extractor \
  --debug