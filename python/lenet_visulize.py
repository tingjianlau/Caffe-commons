from pylab import *
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import caffe

# run scripts from caffe root
import os
os.chdir(caffe_root)
# back to examples
os.chdir('examples')

from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
	n = caffe.NetSpec()

	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255), ntop=2)

	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.fc1, in_place=True)
	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss =  L.SoftmaxWithLoss(n.score, n.label)
													     
	return n.to_proto()
															    
with open('mnist/lenet_auto_train.prototxt', 'w') as f:
	f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
													    
with open('mnist/lenet_auto_test.prototxt', 'w') as f:
	f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')

# each output is (batch size, feature dim, spatial dim)
[(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (we'll omit the biases)
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
