#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file import FileProcess
from smote import Smote
import numpy as np 

filename = 'GermanData.csv'

CLASS = 0  



def np_array(data_list):
    data = []
    for d in data_list:
        data.append(list(map(int,d)))
    data = np.array(data)
    return data

# get serveral constants
def data_get():
    f = FileProcess()
    data_0 = []
    data_1 = []
    headers = []
    headers, data = f.file_get_data_row(filename)
    for line in data:
        if line[CLASS] == '0':
            data_0.append(line[1:])
        else:
            data_1.append(line[1:])
    
    ## class 0 to class 1 ratio
    k = len(data_0)/len(data_1)
    print(k)
    return k, data_0, data_1,headers
    
    



if __name__ == '__main__':
    n = 0
    smote_class = '0'
    data = []
    
    k, data_0, data_1, headers = data_get()

    if k >= 2:
        n = int(k - 1)
        data = data_1
        smote_class = '1'
    elif 1/k >=2:
        n = int(1/k - 1)
        data = data_0
        smote_class = '0'
        
    #data = data[0:5]
    data = np_array(data)
    print(data)

    s=Smote(data,N=100)
    s = (s.over_sampling())
    s = s.tolist()
    
    smote = []
    for line in s:
        l = [0] +  line
        for i in range(1, len(line)+1):
            l[i] = int(line[i-1] + 0.5)
        print(l)
            
        smote.append(l)
    
    f = FileProcess()
    headers, d = f.file_get_data_row(filename)
    
    smote_data = smote + d
    
    f =  FileProcess()
    f.write_csv('smote_data.csv',  headers, smote_data)
    
        

'''
a = []
for d in data_tmp:
    a.append(list(map(int,d)))






a = np.array(a)
print(data_tmp)
s = Smote(a,N=100)
print(s.over_sampling())






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

'''

