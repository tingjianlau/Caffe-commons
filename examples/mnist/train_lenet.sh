#!/usr/bin/env sh

./build/tools/caffe.bin train --solver=examples/mnist/lenet_solver.prototxt --gpu={0,1,2,3}
# | tee -a log_mnist
