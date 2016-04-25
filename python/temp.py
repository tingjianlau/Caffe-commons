import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/usr/local/lib/python2.7/site-packages/')
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2
