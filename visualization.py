import numpy as np
import matplotlib.pyplot as plt
caffe_root = 'home/james/caffe/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cap'] = 'gray'
net = caffe.Classifier(caffe_root+'examples/mnist/lenet_train_test.prototxt',caffe_root+'examples/mnist/lenet_iter_5000.caffemodel')
