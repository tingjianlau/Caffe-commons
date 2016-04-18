#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/12306
DATA=data/12306
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/12306_train_lmdb \
  $DATA/12306_mean.binaryproto

echo "Done."
