import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import caffe
caffe_root= '../'
import os,sys
os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
sys.path.append('usr/local/lib/python2.7/site-packages/')
im = caffe.io.load_image('examples/images/cat.jpg')
print im.shape
plt.imshow(im)
plt.axis('off')
