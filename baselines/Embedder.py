'''
create Word Embedding and user Embedding
'''

import tensorflow as tf
import numpy as np

def WordEmbedder(vocab_size=None, hparams=None):
	with tf.name_scope('WordEmbedding'):
		np.random.seed(10)
		dim = hparams["dim"]
		coorall = np.random.randint(-100,100,(vocab_size,dim))/200.0
		with tf.variable_scope('location_embedding',reuse=tf.AUTO_REUSE): #,reuse=True
			embedding = tf.get_variable('word_embeddings', [vocab_size, dim], initializer = tf.constant_initializer(coorall,dtype=tf.float32)) #trainable=False
			print("vocab size:", vocab_size)
			return embedding

def LocEmbedder(vocab_size=None, coordinate= None, hparams=None):
	np.random.seed(10)
	bias = np.random.randint(-10,10,(8,2))/100.0 
	#base = np.tile(np.array([116.3,39.9]),(8,1))
	#base = np.array([[115.91,39.67],[115.99,39.90],[115.88,40.18],[116.35,40.19],[116.83,40.19],[116.82,39.95],[116.79,39.66],[116.39,39.62]])
	result = bias #+ base #random initial coordinates for special signs
	coordinate[:,0] = coordinate[:,0]-np.min(coordinate[:,0])
	coordinate[:,1] = coordinate[:,1]-np.min(coordinate[:,1])
	coorall = np.vstack((result,coordinate))
	print('init:',coorall[:12,:5])
	#print(coorall.shape) 10664x2
	
	dim = hparams["dim"]
	con1d = tf.get_variable("W",[2, dim],initializer = tf.random_uniform_initializer(minval = -1, maxval = 1))
	embedding = tf.matmul(tf.constant(coorall,dtype=tf.float32),con1d)
	con1b = tf.get_variable("B",[1, dim],initializer = tf.random_uniform_initializer(minval = -1, maxval = 1))
	con1bs = tf.tile(con1b,multiples=[len(coorall), 1])
	embedding = embedding + con1bs
	
	print("vocab size:", vocab_size, 'word size:', coorall.shape , 'emnedding size:', embedding.shape)
	return embedding
