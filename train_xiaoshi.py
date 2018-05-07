# train GOTURN model

import numpy as np 
import tensorflow as tf 



from goturn_config import cfg

from PIL import Image
import time
from model import GOTURNModel

#data pipeline
#filename_q = tf.train.string_input_producer([cfg.TRAIN_FILE])

#NUM_CHANNELS = 3


def read_file(train_file):
	print train_file
	target_list = []
	search_list = []
	label_list = []
	with open(train_file, 'rb') as f:
		content = f.readlines()
		for line in content:
			target_list.append(line.split(',')[0])
			search_list.append(line.split(',')[1])
			label_list.append(line.split(',')[2:])

	target_tensor = tf.convert_to_tensor(target_list)
	search_tensor = tf.convert_to_tensor(search_list)
	label_tensor = tf.convert_to_tensor(label_list)
	return target_tensor, search_tensor, label_tensor

#reader = tf.WholeFileReader()
def data_pipeline(train_file, batch_size):

	target_list, search_list, label_list = read_file(train_file)
	data_queue = tf.train.slice_input_producer([target_list, search_list, label_list], shuffle=False)
	#print data_queue[0]
	#reader = tf.WholeFileReader()
	#target_name, target_val = reader.read(data_queue[0])
	target_val = tf.read_file(data_queue[0])
	target_img = tf.image.decode_jpeg(target_val, channels=cfg.NUM_CHANNELS)
	target_img = tf.image.resize_images(target_img,(cfg.IMAGE_WIDTH, cfg.IMAGE_WIDTH))
	#search_name, search_val = reader.read(data_queue[1])
	search_val = tf.read_file(data_queue[1])
	search_img = tf.image.decode_jpeg(search_val, channels=cfg.NUM_CHANNELS)
	search_img = tf.image.resize_images(search_img,(cfg.IMAGE_WIDTH, cfg.IMAGE_WIDTH))
	label = data_queue[2]
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3*batch_size
	target_batch, search_batch, label_batch = tf.train.shuffle_batch([target_img, search_img, label],
													batch_size=batch_size, capacity=capacity,
													min_after_dequeue = min_after_dequeue, 
													num_threads=4)
	return target_batch, search_batch, label_batch


with tf.Session() as sess:
	target_batch, search_batch, label_batch = data_pipeline(cfg.TRAIN_FILE, cfg.BATCH_SIZE)

	model = GOTURNModel(cfg)
	model.build(True)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	start = time.time()
	for i in range(200):
		print "Iter: ",i
		target_data, search_data, label_data = sess.run([target_batch, search_batch, label_batch])
		feed_dict = {model.target_patch: target_data, model.search_patch: search_data, model.labels:label_data}
		sess.run(model.train_op, feed_dict=feed_dict)

	
	coord.request_stop()
	coord.join(threads)
	eclipsed = time.time() - start
	print "time spend: ",eclipsed	




	#reader = tf.TextLineReader()
	#key, value = reader.read(filename_queue)



