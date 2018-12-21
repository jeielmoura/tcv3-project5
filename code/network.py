import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

from constants import *
from data import *

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Create a training graph that receives a batch of images and their respective labels and run a      #
#         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
#         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default() :
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	inp = tf.placeholder(tf.float32, shape = (None, INPUT_SIZE))
	y = tf.placeholder(tf.int64, shape = (None,))
	batch_size = tf.placeholder(tf.int64, shape = (1,)) 


	y_one_hot = tf.one_hot(y, 2)
	learning_rate = tf.placeholder(tf.float32)
	
	
	with tf.variable_scope('generator') :
		gen = tf.layers.dense(inp, 256, activation=tf.nn.relu)
		print('gen FC:', gen.shape)
		gen = tf.reshape(gen, [-1, 16, 16, 1])
		gen = tf.layers.conv2d_transpose(gen, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		gen = tf.layers.conv2d_transpose(gen, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print('gen CV:', gen.shape)

	
	batch = tf.concat([gen, X], 0)

	with tf.variable_scope('discriminator') :
		dis = tf.layers.conv2d(batch, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		dis = tf.layers.conv2d(dis, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print('dis CV:', dis.shape)
		dis = tf.reshape(dis, [-1, 16*16])
		dis = tf.layers.dense(dis, 256, activation=tf.nn.relu)
		print('dis FC:', dis.shape)
		
		
	out = tf.layers.dense(dis, 2, activation=tf.nn.sigmoid)


	loss = tf.reduce_mean( tf.reduce_sum( (y_one_hot - out)**2 ) )


	gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
	dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

	gen_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=gen_vars)
	dis_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=dis_vars)

	result = tf.argmax(dis, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))


# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Run one training epoch using images in X_train and labels in y_train.                              #
# ---------------------------------------------------------------------------------------------------------- #
def training_epoch(session, lr, epoch):
	batch_list = np.random.permutation(len(X_data))

	start = time.time()
	train_loss1, train_loss2 = 0, 0
	train_acc1, train_acc2 = 0, 0

	for j in range(0, len(X_data), HALF_BATCH_SIZE):
		if j+HALF_BATCH_SIZE > len(X_data):
			break

		X_batch = X_data.take(batch_list[j:j+HALF_BATCH_SIZE], axis=0)
		
		y_batch_true = np.ones(HALF_BATCH_SIZE, dtype=np.int64)
		y_batch_false = np.zeros(HALF_BATCH_SIZE, dtype=np.int64)

		inp_batch = np.random.normal( size=(HALF_BATCH_SIZE,INPUT_SIZE) )

		y_batch	= np.concatenate( (y_batch_true, y_batch_false), axis=0)

		
		ret1 = session.run( [dis_op, loss, correct],
							feed_dict = {	
											X: X_batch,
											y: y_batch,
											inp: inp_batch,
											learning_rate: lr
										}
						  )

		inp_batch = np.random.normal( size=(HALF_BATCH_SIZE,INPUT_SIZE) )

		ret2 = session.run( [gen_op, loss, correct],
							feed_dict = {
											X: np.empty( shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) ),
											y: y_batch_true,
											inp: inp_batch,
											learning_rate: lr
										}
						  )

		train_loss1 += ret1[1]*HALF_BATCH_SIZE*2
		train_loss2 += ret2[1]*HALF_BATCH_SIZE


		train_acc1 += ret1[2]
		train_acc2 += ret2[2]


	print('Epoch:'+ epoch, 'LR:' + str(lr),  'Time:'+str(time.time()-start) )
	pass_size = ( len(X_data)-len(X_data)%HALF_BATCH_SIZE )
	print('\tacc1:' + str(train_acc1/pass_size), 'loss1:' + str(train_loss1/pass_size) )
	pass_size = HALF_BATCH_SIZE
	print('\tacc2:' + str(train_acc2/pass_size), 'loss2:' + str(train_loss2/pass_size) )

	return train_acc2/pass_size, train_loss2/pass_size

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Training loop and execution.																		 #
# ---------------------------------------------------------------------------------------------------------- #	
def train_model (model_version) :
	global X_data
	global NUM_EPOCHS
	global LEARNING_RATES, LEARNING_RATE_DECAY


	LEARNING_RATES_ARRAY = np.array(LEARNING_RATES)

	MODEL_PATH = MODEL_FOLDER + '/' + 'model' + model_version
	
	with tf.Session(graph = graph) as session:
		session.run(tf.global_variables_initializer())

		best_val_acc = -1.
		best_val_loss = -1.

		saver = tf.train.Saver( var_list= [v for v in tf.trainable_variables() if v.name.startswith('generator')] )

		
		for epoch in range(NUM_EPOCHS) :
			
			for i in range( len(LEARNING_RATES_ARRAY) ) :
				LR = LEARNING_RATES_ARRAY[i]

				val_acc, val_loss = training_epoch(session, LR, str(epoch) + '.' + str(i) )



				better = (best_val_acc <= val_acc and best_val_loss >= val_loss)
				better = better or (best_val_acc < 0. and best_val_loss < 0.)

				if better :
					best_val_acc = val_acc
					best_val_loss = val_loss
					saver.save(session, MODEL_PATH)
					print()

			LEARNING_RATES_ARRAY = np.array(LEARNING_RATES)*LEARNING_RATE_DECAY
			
			save_imgs = session.run (
									[gen],
									feed_dict = {
													X: np.empty( shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) ),
													y: np.empty( shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) ),
													inp: np.random.normal( size=(HALF_BATCH_SIZE,INPUT_SIZE) ),
												}
								).reshape( (SAVE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) )
			
			for i in range( len(save_imgs) ) :
				cv2.imwrite (RESULT_FOLDER + '/' + str(epoch) + '-' + str(i) + '.png', save_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH) )