#goturn model
import tensorflow as tf 
import numpy as np 
#from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.slim as slim
#from tensorflow.python.ops import init_ops


class GOTURNModel:
	def __init__(self, cfg):
		self.cfg = cfg
		self.input()


	def build(self, train=True):

		self.optimizer = self.optimizer()
		global_step = tf.Variable(0, trainable=False, name="global_step")
		self.logits = self.forward(train=train)
		self.loss = self.loss(self.logits, self.labels)
		self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

	def input(self):
		cfg = self.cfg
		batch_size = cfg.BATCH_SIZE
		self.target_patch = tf.placeholder(tf.float32, [batch_size, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
		self.search_patch = tf.placeholder(tf.float32, [batch_size, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
		self.labels = tf.placeholder(tf.float32, [batch_size, 4])


	def forward(self, train=False):
		# target object
		net_t = goturn_conv(self.target_patch, 96, 11,4,'VALID', train, 1, 'target_1')
		net_t = slim.max_pool2d(net_t, 3, 2)
		net_t = tf.nn.lrn(net_t, depth_radius=5, bias=1, alpha=0.0001, beta=0.75, name='tnorm1')

		net_t = goturn_conv(net_t, 256, 5, 1, 'SAME', train, 2,'target_2')
		net_t = slim.max_pool2d(net_t, 3, 2)
		net_t = tf.nn.lrn(net_t, depth_radius=5, bias=1, alpha=0.0001, beta=0.75, name='tnorm2')

		net_t = goturn_conv(net_t, 384, 3, 1, 'SAME', train, 1, 'target_3')
		net_t = goturn_conv(net_t, 384, 3, 1, 'SAME', train, 2, 'target_4')
		net_t = goturn_conv(net_t, 256, 3, 1, 'SAME', train, 2, 'target_5')
		net_t = slim.max_pool2d(net_t, 3, 2)

		# search patch
		net_s = goturn_conv(self.search_patch, 96, 11,4,'VALID', train, 1, 'search_1')
		net_s = slim.max_pool2d(net_s, 3, 2)
		net_s = tf.nn.lrn(net_s, depth_radius=5, bias=1, alpha=0.0001, beta=0.75, name='snorm1')

		net_s = goturn_conv(net_s, 256, 5, 1, 'SAME', train, 2,'search_2')
		net_s = slim.max_pool2d(net_s, 3, 2)
		net_s = tf.nn.lrn(net_s, depth_radius=5, bias=1, alpha=0.0001, beta=0.75, name='snorm2')

		net_s = goturn_conv(net_s, 384, 3, 1, 'SAME', train, 1, 'search_3')
		net_s = goturn_conv(net_s, 384, 3, 1, 'SAME', train, 2, 'search_4')
		net_s = goturn_conv(net_s, 256, 3, 1, 'SAME', train, 2, 'search_5')
		net_s = slim.max_pool2d(net_s, 3, 2)

		net = tf.concat([net_t, net_s], axis=1)

		#fc6
		net = tf.layers.flatten(net)
		net = slim.fully_connected(net, 4096, biases_initializer=tf.ones_initializer(), trainable=train)
		net = tf.nn.relu(net)
		net = slim.dropout(net, 0.5, is_training=train)

		#fc7
		net = tf.layers.flatten(net)
		net = slim.fully_connected(net, 4096, biases_initializer=tf.ones_initializer(), trainable=train)
		net = tf.nn.relu(net)
		net = slim.dropout(net, 0.5, is_training=train)


		#fc7-2
		net = tf.layers.flatten(net)
		net = slim.fully_connected(net, 4096, biases_initializer=tf.ones_initializer(), trainable=train)
		net = tf.nn.relu(net)
		net = slim.dropout(net, 0.5, is_training=train)


		#fc-8
		net = slim.fully_connected(net, 4, biases_initializer=tf.zeros_initializer(), trainable=train)

		return net

	def loss(self, logits, labels):
		loss =  tf.losses.mean_squared_error(labels, logits)
		return loss

	def optimizer(self):
		cfg = self.cfg
		return tf.train.AdamOptimizer(cfg.LEARNING_RATE)

	#def train_step(self, sess):


	#def test(self):


def goturn_conv(inputs, num_outputs, kernel_size, stride, padding, train, groups, name):
	if isinstance(kernel_size, int):
		kernel_size = [kernel_size, kernel_size]
	if groups == 1:
		net = slim.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, trainable=train, scope=name)
	elif groups == 2:
		split_channel_num = inputs.shape[3]/2
		#print "split_channel_num ",split_channel_num
		split_filter_num = num_outputs/2
		#print "split_filter_num ",split_filter_num
		weight_name = name+"_weight"
		
		weight_shape = [kernel_size[0],kernel_size[1],split_channel_num, num_outputs]	
		weights = tf.get_variable(weight_name,shape=weight_shape, initializer=tf.contrib.layers.xavier_initializer(),trainable=train)
		
		conv_res1 = tf.nn.conv2d(inputs[:,:,:,:split_channel_num], weights[:,:,:,:split_filter_num], [1,stride,stride,1],padding=padding)
		conv_res2 = tf.nn.conv2d(inputs[:,:,:,split_channel_num:], weights[:,:,:,split_filter_num:], [1,stride,stride,1],padding=padding)

		if groups == 1:
			bias_num = 0
		else:
			bias_num = 1
		bias_name = name+"_bias"
		bias = tf.get_variable(bias_name, shape=[num_outputs], initializer=tf.constant_initializer(bias_num), trainable=train)
		net = tf.nn.bias_add(tf.concat([conv_res1, conv_res2], axis=3), bias)
		net = tf.nn.relu(net)

	return net


		