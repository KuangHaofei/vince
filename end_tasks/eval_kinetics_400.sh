#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_kinetics_400"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python run_end_task_eval.py \
  --base-logdir ${BASE_LOG_LOCATION} \
  --tensorboard-dir "tensorboard/eval"\
  --title ${TITLE} \
  --description r18-b-256-q-65536-fsize-64-vid-ibc-4-kinetics \
  --solver EndTaskKinetics400Solver \
  --dataset Kinetics400Dataset \
  --data-path /home/ubuntu/drive3/kinetics400_30fps_frames/ \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 0 \
  --backbone ResNet18 \
  --no-save \
  --disable-dataloader \
  --freeze-feature-extractor \
  --num-workers 40 \
  --num-frames 10 \
  --batch-size 64 \
  --input-width 224 \
  --input-height 224 \
  --base-lr 0.01 \
  --end-task-classifier-num-classes 400 \
  --use-apex