#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from file import FileProcess

filename = 'default of credit card clients.csv'



##
def get_standard_data_for_cloumn(filename, header):
    f =  FileProcess()
    d = f.file_get_data(filename, header)
    d = list(map(int, d)) 
    max_d = max(d)
    min_d = min(d)
    #print(d)
    if max_d == min_d:
        return d
    
    if header == 'class':
        return d
    for i in range(len(d)):
        d[i] = int((d[i] - min_d)*(9/(max_d - min_d)))
        
    #print(d)
    #print(max_d, min_d)
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
    return elem[-1]

def data_sort(result_list):
        #sort_list = []
        result_list.sort(reverse=True, key=__data_takeSecond)         
        return result_list
   
    
def get_train_val_test(data):
    
    for i in range(len(data)):
        if data[i][-1] == 0:
            len1 = i + 1
            break
     
    len2 = len(data) - len1       
    
    
    idx1 = int(10 * len1/13)
    idx2 = int(idx1 + len1/13)
    
    data_train1 = data[0:idx1]
    data_val1 = data[idx1:idx2]
    data_test1 = data[idx2:len1]
    #print(idx1, idx2 - idx1, len1-idx2)

    
    idx1 = len1 + int(10 * len2/13)
    idx2 = int(idx1 + len2/13)
    
    data_train0 = data[len1:idx1]
    data_val0 = data[idx1:idx2]
    data_test0 = data[idx2:-1]
    #print(idx1-len1, idx2 - idx1, len(data)-idx2)
    
    data_train = data_train1 + data_train0
    data_val = data_val1 + data_val0
    data_test = data_test1 + data_test0
    
    return data_train,data_val,data_test
  
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
