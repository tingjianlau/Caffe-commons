#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/12306/solver.prototxt --gpu={0,1,2,3}
