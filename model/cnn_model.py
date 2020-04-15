# coding: utf-8

import tensorflow as tf

import tensorflow.keras.backend as K


class CNNConfig(object):

    combination_feature = 8 
    
    seq_length = 32  
    num_classes = 2  
    num_filters = 64  
    kernel_size = 2  
   

    hidden_dim = 64  

    dropout_keep_prob = 1 
    learning_rate = 1e-3 

    batch_size = 128  
    num_epochs = 10000  

    print_per_batch = 20  
    save_per_batch = 5 


class CNN(object):

    def __init__(self, config):
        self.config = config

        
        self.input_x = tf.placeholder(tf.float32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        
        
    def focal_loss(self, pred, y, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         pred: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         y: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        zeros = tf.zeros_like(pred, dtype=pred.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

        return tf.reduce_sum(per_entry_cross_ent)
        



    def cnn(self):
        
        print('x_shap1',self.input_x.shape)
         
        ##feature combination layer
        with tf.name_scope("extra_feature"):
            cob_fe = tf.layers.dense(self.input_x, self.config.combination_feature, name='orfc')
            new_input = tf.concat([cob_fe, self.input_x], 1)
             
        
        
        
        with tf.name_scope("feature_reduction"):
            
            fc = tf.layers.dense(new_input, self.config.hidden_dim, name='fc0')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        
        
        ##batch_size, max_time, input_size
        #new_input = new_input.reshape([X_test.shape[0], 18, 128])
        new_input = tf.expand_dims(fc, axis=2)
        

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(new_input, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  
            ## add for auc
            self.softmax_score = tf.nn.softmax(self.logits)

        with tf.name_scope("optimize"):
            # 
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #self.loss = tf.reduce_mean(cross_entropy)

            cross_entropy = self.focal_loss(self.logits,self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            #
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
