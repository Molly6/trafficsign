#!/usr/bin/env bash

gpu=4
WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/

# train
python3 tools/train.py -n ${gpu} -b 2 \
  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d /data/Dataset/ \
  -w weights/faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl

# test

# 1X
python3 tools/test.py -n ${gpu} -se 11 \
  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d /data/Dataset/

# 2X
python3 tools/test.py -n ${gpu} -se 23 \
  -f configs/faster_rcnn_res50_800size_trafficdet_demo.py -d /data/Dataset/