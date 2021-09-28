#!/usr/bin/env bash

gpu=2
WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/

# train
python3 tools/train.py -n ${gpu} -b 2 \
  -f configs/fcos_res50_800size_trafficdet_demo.py -d . \
  -w weights/fcos_res50_coco_3x_800size_42dot2_b16f9c8b.pkl

# test

# 1X
python3 tools/test.py -n ${gpu} -se 11 \
  -f configs/fcos_res50_800size_trafficdet_demo.py -d .

# 2X
python3 tools/test.py -n ${gpu} -se 23 \
  -f configs/fcos_res50_800size_trafficdet_demo.py -d .