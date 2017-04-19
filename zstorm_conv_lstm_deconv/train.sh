#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=zstorm_conv_lstm_deconv/solver.prototxt \
  --gpu=2,3 \
  2>&1 | tee zstorm_conv_lstm_deconv/train.log
