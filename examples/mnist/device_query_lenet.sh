#author: tingjianlau
#!/usr/bin/env sh

./build/tools/caffe.bin device_query --gpu={0,1,2,3} 
# | tee -a log_mnist
