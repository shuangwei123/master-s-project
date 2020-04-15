#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time

from sys import path
path.append('./data')
path.append('./model')

from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_attention import CNNConfig, CNN
from rnn_model import RNNConfig, RNN

from sklearn.metrics import roc_auc_score
import data_loader
from data_loader import  read_category, batch_iter, process_file, build_vocab

from sklearn.metrics import brier_score_loss



data_list = ['default', 'German','Japan','Australian','GiveMeSomeCredit',
             'lpetrocelli']



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 
    tensorboard_dir = 'tensorboard/cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    #
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 
    start_time = time.time()
    x_train, y_train = process_file(train_dir, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, cat_to_id, config.seq_length)

    #
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  #
    best_acc_val = 0.0  # 
    last_improved = 0  # 
    require_improvement = 10000  #

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                
                #print(feed_dict)
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                #
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict) 
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 
        if flag:  # 
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('save_path',save_path)
    saver.restore(sess=session, save_path=save_path) 

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.4}, Test Acc: {1:>7.4%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  
    
    auc_score = [] 
    for i in range(num_batch): 
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
        ##add for auc
        softmax_score = session.run(model.softmax_score,feed_dict=feed_dict)
        for line in softmax_score:
            auc_score.append(line[1])
       
        
    auc_score = np.array(auc_score)
    #auc = round(metrics.auc(y_test_cls,auc_score),4)
    auc = round(roc_auc_score(y_test_cls, auc_score),4)
    print('Test Auc:', str(auc*100)+'%')
    
    brier_score = brier_score_loss(y_test_cls, auc_score) 
    print('Brierscore',brier_score)


    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    if data_dir not in data_list:
        print('the data not exist.')
    
    base_dir = 'data/%s'%data_dir
    train_dir = os.path.join(base_dir, 'data_train.csv')
    test_dir = os.path.join(base_dir, 'data_test.csv')
    val_dir = os.path.join(base_dir, 'data_val.csv')
    save_dir = 'checkpoints/cnn'
    save_path = os.path.join(save_dir, 'best_validation')  
    

    config = CNNConfig()
    categories, cat_to_id = read_category()
    model = CNN(config)
    if sys.argv[2] == 'train':
        train()
    else:
        test()
