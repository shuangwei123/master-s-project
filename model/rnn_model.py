#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class RNNConfig(object):

    # hyper parameters
    combination_feature = 0  

    embedding_dim = 64      
    seq_length = 128        
    num_classes = 2         
    vocab_size = 5000      

    num_layers= 2          
    hidden_dim = 128        
    rnn = 'gru'             

    dropout_keep_prob = 0.8 
    learning_rate = 1e-3    

    batch_size = 64        
    num_epochs = 1000          

    print_per_batch = 100
    save_per_batch = 10     


class RNN(object):

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):

        def lstm_cell():   
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        
        def gru_cell(): 
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): 
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)


        with tf.name_scope("extra_feature"):
            cob_fe = tf.layers.dense(self.input_x, self.config.combination_feature, name='orfc')
            new_input = tf.concat([cob_fe, self.input_x], 1)
           
        
        
        ##batch_size, max_time, input_size
        '''
        
        with tf.device('/cpu:0'):
            #print([self.config.vocab_size, self.config.embedding_dim])
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable('embedding', [self.config.vocab_size + self.config.combination_feature, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, new_input)
        '''
        #new_input = new_input.reshape([X_test.shape[0], 18, 128])
        new_input = tf.expand_dims(new_input, axis=2)
        

        with tf.name_scope("rnn"):
            
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=new_input, dtype=tf.float32)
            last = _outputs[:, -1, :]  

        with tf.name_scope("score"):
            
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

         
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.softmax_score = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 

        with tf.name_scope("optimize"):
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
