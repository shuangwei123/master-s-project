#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:55:55 2020

@author: arno
"""

from file import FileProcess
import random
import math

filename = 'smote_data.csv'

label_list = ['Duration of Credit (month)','Credit Amount','Age (years)']


def deal_with_NA(d):
    s = 0
    for i in d:
        if i != 'NA':
            s += int(i)
    aver = int(s/len(d))
    for i in range(len(d)):
        if d[i] == 'NA':
            d[i] = aver
    return d
    
        


##
def get_standard_data_for_cloumn(filename, header):
    
    
    f =  FileProcess()
    d = f.file_get_data(filename, header)
    try:
        d = deal_with_NA(d)
        d = list(map(int, d)) 
    except:
        d = list(map(float, d)) 
   
    max_d = max(d)
    min_d = min(d)
    if max_d == min(d):
        return d
    
    #if header not in label_list:
    #    return d


    if header == 'Creditability':
        return d
    for i in range(len(d)):
        #if d[i] == -1:
            #d[i] = 99
        
        x = len(str(d[i]))
        d[i] = d[i]/(math.pow(10, x))
        d[i] = round(d[i],1)
        #d[i] = int((d[i] - min_d)*(9/(max_d - min_d)))
        
    
    return d

def data_transpose(r):
    data = []
    for i in range(len(r[0])):
        data_temp = []
        for j in range(len(r)):
            data_temp.append(r[j][i])
        data.append(data_temp)
    return data


def __data_takeSecond( elem):
    return elem[0]

def data_sort(result_list):
        #sort_list = []
        result_list.sort(reverse=True, key=__data_takeSecond)         
        return result_list
   
    
def get_train_val_test(data):
    
    for i in range(len(data)):
        if data[i][0] == 0:
            len1 = i + 1
            break
     
    len2 = len(data) - len1       
    
    
    idx1 = int(16 * len1/25)
    idx2 = int(idx1 + 4*len1/25)
    
    data_train1 = data[0:idx1]
    data_val1 = data[idx1:idx2]
    data_test1 = data[idx2:len1]
    #print(data_val1)

    
    idx1 = len1 + int(16 * len2/25)
    idx2 = int(idx1 + 4*len2/25)
    
    data_train0 = data[len1:idx1]
    data_val0 = data[idx1:idx2]
    data_test0 = data[idx2:-1]
    #print(idx1-len1, idx2 - idx1, len(data)-idx2)
    #data_train1 = expand_val(data_train1,15)
    #data_val1 = expand_val(data_val1,15)
    #data_test1 = expand_val(data_test1,15)
    
    
    data_train = data_train1 + data_train0#[0:len(data_train1)]
    data_val = data_val1 + data_val0#[0:len(data_val1)]
    data_test = data_test1 + data_test0#[0:len(data_test1)]
    
    return data_train,data_val,data_test


###用例扩充
def expand_val(data,expand_num):
    expand_data = []
    for i in range(0, len(data) - expand_num):
        expand_data.append(data[i])
        #print('---',data[i])
        for j in range(1, expand_num+1):
            line1 = data[i]
            line2 = data[i+j]
            line = []
            for k in range(len(line1)):
                if line1[k] == 199:
                    result = 199
                else:
                    result = int((line2[k] + line1[k])*0.5)
                if result < 0:
                    result = 0
                if result >99:
                    result =99
                line.append(result)
            
            expand_data.append(line)
            #print(line)
    for i in range(len(data) - expand_num, len(data)):
        expand_data.append(data[i])
        
        
    return expand_data
        
                
            
            
            
        
    
  
#f.write_csv('data_train.csv',  headers, data_train)
def data_stard_write():
    data_all = []
    f =  FileProcess()
    headers = f.file_get_headers(filename) 
    for header in headers:
        d = get_standard_data_for_cloumn(filename, header)
        data_all.append(d)
        
    data_all = data_transpose(data_all)
    data_all = data_sort(data_all)
    data_train,data_val,data_test = get_train_val_test(data_all)
    f.write_csv('data_train.csv',  headers, data_train)
    f.write_csv('data_val.csv',  headers, data_val)
    f.write_csv('data_test.csv',  headers, data_test)

data_stard_write()
#get_standard_data_for_cloumn(filename, 'RevolvingUtilizationOfUnsecuredLines')

#print(float(random.random()))

     
    
