#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#from keras.preprocessing import sequence
#from keras.datasets import imdb
#from matplotlib import pyplot as plt
#import pandas as pd
 
from tensorflow.keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
 
import tensorflow as tf 



class Insignificant_Attention(Layer):
 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Insignificant_Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        
        #inputs.shape = (batch_size, time_steps, seq_len)
        print(input_shape[2])
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2],self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
 
        super(Insignificant_Attention, self).build(input_shape)  
 
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
 
        print("WQ.shape",WQ.shape)
 
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
 
 
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        

        ##add for feature reduction
        b = tf.reduce_sum(QK,reduction_indices=2)

        c = tf.argmin(b,1)
        c = tf.to_int32(c)

        QK1  = tf.slice(QK, [0, 0, 0], [-1, c[0], -1])
        QK2  = tf.slice(QK, [0, c[0]+1, 0], [-1, -1, -1])
        OK_0 = tf.slice(QK, [0, 0, 0], [-1, 1, -1])
        QK = tf.concat([QK1, QK2],1)
        
        
        QK = QK / (64**0.5)
        QK = K.softmax(QK)
        
        
        shape = tf.shape(OK_0)
        #print(shape, shape[0],shape[1])
        
        
        ##add for feature reductuins
        QK1 = tf.slice(QK, [0, 0, 0], [-1, c[0], -1])
        QK2 = tf.slice(QK, [0, c[0], 0], [-1, -1, -1])
        OK_0 = tf.zeros(shape, dtype = tf.float32)
        QK = tf.concat([QK1, OK_0, QK2],1)
        

        print("OK_0.shape",OK_0.shape)
 
        V = K.batch_dot(QK,WV)
        
        print("v.shape",V.shape)
 
        return V
 
    def compute_output_shape(self, input_shape):
 
        return (input_shape[0],input_shape[1],self.output_dim)



def insignificantAttention(inputs, attention_size, time_major=False, return_alphas=False):


    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
