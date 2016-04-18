#!/usr/local/bin/python

from os import system
import sys

if __name__ == '__main__':
	#if sys.argv[1].strip == 'imagenet':
	#	tool = './build/examples/cpp_classification/classification.bin'
	#	solver = './models/bvlc_reference_caffenet/deploy.prototxt'
	#	model = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	#	mean = './data/ilsvrc12/imagenet_mean.binaryproto'
	#	label = './data/ilsvrc12/synset_words.txt'
	#	image = './examples/images/cat.jpg'
	#	print 'imagenet test'
	#else:
	if len(sys.argv) < 2:
		print 'Usage: python classification.py [/path/to/data/val]'
	else:
		num = sys.argv[2].strip()
		tool = './build/examples/cpp_classification/classification.bin'
		solver = './examples/12306/deploy.prototxt'
		model = './models/12306/caffenet_train_iter_%d0000.caffemodel' % int(num)
		mean = './data/12306/12306_mean.binaryproto'
		label = './data/12306/synset_words.txt'
		image = sys.argv[1].strip()
		#print '12306 test'
		
		command = tool + ' ' + solver + ' ' + model + ' ' + mean + ' ' + label + ' ' + image
		system(command)
