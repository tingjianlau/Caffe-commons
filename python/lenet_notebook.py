# encoding:utf-8
#--------------------1. Setup -------------------#
### 1.1) Set up the Python environment ###
from pylab import *

### 1.2) Import caffe
caffe_root = '../'

import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import caffe

### 1.3) Download the data and create the datasets ###
# run scripts from caffe root
import os
os.chdir(caffe_root)
# Download data
if not os.path.isfile('data/mnist/train-images-idx3-ubyte'):
	print('Downloading...')
	command = 'data/mnist/get_mnist.sh'
	os.system(command)
else:
	print 'the data has downloaded'
# Prepare data
if not os.path.exists('examples/mnist/mnist_train_lmdb/'):
	print('Creating...')
else:
	print('the datasets has created')

#!examples/mnist/create_mnist.sh
# back to examples
os.chdir('examples')


#--------------------2. Creating the net -------------------#
### 2.1) create the net using python ###
# omit

### 2.2) Load the net prototxt file ###
net_prototxt_file = 'mnist/lenet_auto_train.prototxt'
command = 'cat ' + net_prototxt_file
#os.system(command)

### 2.3) Load the solver prototxt file ###
solver_prototxt_file = 'mnist/lenet_auto_solver.prototxt'
command = 'cat ' + solver_prototxt_file
#os.system(command)


#--------------------3. Loading and checking the solver -------------------#
### 3.1) picking a device and load the solver ###
caffe.set_device(0)
caffe.set_mode_gpu()

# 加载solver并创建训练和测试网络
solver = None # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(solver_prototxt_file)

### 3.2) checking the dimensions of the intermediate features(blobs) and parameters
print('打印各层的激活值的shape信息:')
for k, v in solver.net.blobs.items():
	print(k,v.data.shape)

print('打印各层的可学习参数的shape信息:')
for k, v in solver.net.params.items():
	print(k, v[0].data.shape, v[1].data.shape)

### 3,3) Runing a forward pass on the train and test to check that everyting is loaded as we expected
solver.net.forward()
solver.test_nets[0].forward() # test net (there can be more than one)

print('After one forward pass, the loss is: %f' % solver.net.blobs['loss'].data)

### 3.4) Using a trick to tile the first eight images ###
imshow(solver.net.blobs['data'].data[0:8,0].transpose(1,0,2).reshape(28,8*28), cmap='gray'); axis('off')
#plot(solver.net.blobs['data'].data[0:8,0].transpose(1,0,2).reshape(28,8*28))
#savefig('./temp.jpg')
print 'train labels:', solver.net.blobs['label'].data[:8]

imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]


#--------------------4. Stepping the sovler -------------------#
### Taking one step ###
# omit


#--------------------5. Writing a custom training loop -------------------#
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
	
	solver.step(1) # SGD by caffe

	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data

	# store the output on the first test batch
	# (start the forward pass at conv1 to avoid loading new data)
	solver.test_nets[0].forward(start='conv1')
	output[it] = solver.test_nets[0].blobs['score'].data[:8]

	# run a full test every so often
	# (Caffe can also do this for us and write to a log, but we show here
	#  how to do it directly in Python, where more complicated things are easier.)
	if it % test_interval == 0:
		print 'Iteration', it, 'testing...'
		correct = 0
		for test_it in range(100):
			solver.test_nets[0].forward()
			correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)== solver.test_nets[0].blobs['label'].data)
			test_acc[it // test_interval] = correct / 1e4

### Plotting the train loss and test accuracy ###
__, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
savefig('./loss_accuracy.jpg')

