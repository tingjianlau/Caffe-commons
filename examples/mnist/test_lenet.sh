#author: tingjianlau
#!/usr/bin/env sh

./build/tools/caffe.bin test --model=examples/mnist/lenet_train_test.prototxt --weights=examples/mnist/lenet_iter_10000.caffemodel --gpu={0,1,2,3} --iterations=100
# | tee -a log_mnist
