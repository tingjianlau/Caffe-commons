#!/usr/bin/python
# test under folder data/12306/val/

import os
from os import system

prefix = '/home/tjliu/caffe/'
val = 'data/12306/val'
log = '12306_log'
synset_file = 'data/12306/synset_words.txt'
word_dict = {}
top1_word_dict = {}

def make_dict(syn_file):
	with open(syn_file, 'r') as df:
		for kv in [d.split(' ') for d in df]:
			word_dict[kv[0]] = 0
			top1_word_dict[kv[0]] = 0

if __name__ == '__main__':
	make_dict(prefix + synset_file)
	f = open('12306_result.txt', 'a+')
	command = 'python classification.py %s %d >%s' %((prefix + val), 45, log)
	system(command)
	df = open(prefix + log)
	
	while 1:
		temp = df.readline()
		if not temp:
			break
		imgpath = temp.strip().split(' ')[3];
		category = imgpath.split('/')[7]
		cnt = 5
		while cnt > 0:
			cnt = cnt - 1
			result = df.readline().strip().split('"')[1].split(' ')[0]
			if result == category:
				break
			if cnt == 4:
				top1_word_dict[category] = top1_word_dict[category] + 1
		if result != category:
			#print >> f, '[Warning] class is ' + category + ', but result is ' + result
			#print >> f, 'Testing image is' + imgpath;
			word_dict[category] = word_dict[category] + 1
		for i in range(cnt):
			df.readline()
	total_num = 3400
	top1_err_num = 0
	top5_err_num = 0
	for k in top1_word_dict:
		top1_err_num = top1_err_num + top1_word_dict[k]
		print >> f, 'Class ' + k + ':' + str(top1_word_dict[k]) + ' errors.'
		top1_word_dict[k] = 0
	for k in word_dict:
		top5_err_num = top5_err_num + word_dict[k]
		word_dict[k] = 0
	print >> f, 'Top-1 error rate is %f\nTop-5 error rate is %f' % ((float(top1_err_num) / float(total_num)), (float(top5_err_num) / float(total_num)))
