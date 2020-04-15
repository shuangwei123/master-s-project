# coding: utf-8

import sys
import csv
from collections import Counter

import numpy as np



import tensorflow.contrib.keras as kr
from itertools import combinations



data_dict = {'default':1, 'German':0,'Japan':1,'Australian':1,'GiveMeSomeCredit':0,
             'lpetrocelli':1}

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):

    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)
    
    
def is_number(s):
    try:
        if s=='NaN':
            return False
        float(s)
        return True
    except ValueError:
        return False
    
    
def to_number(s):
    if s == '?':
        return s
    if is_number(s) == False:
        result = 0
        for i in range(len(s)):
            result += (ord(s[i]) - 96)+ i*24
        return str(result)
            
    else:
        return s
    
    
#GNN
def gnn(contents, lists):
    m = lists[0]
    n = lists[1]
    p = lists[2]
    q = lists[3]
    r1 = []
    r2 = []
    for i in range(len(contents)):
        t1 = (contents[i][m] + contents[i][n])/2
        t2 = (contents[i][p] + contents[i][q])/2
        contents[i] = [t1] + [t2] + contents[i] 
        
    return contents
    
            
   
##    
def data_normalization(contents):
    for i in range(len(contents)):
        for j in range(len(contents[0])):
            contents[i][j] = float(contents[i][j])/(len(str(contents[i][j]))*10)
            
    return contents

###data normalization
def data_normalizations(contents):
        
    lens = len(contents[0])
    max_list = [0.0 for i in range(lens)]
    min_list = [999999.0 for i in range(lens)]
    for line in contents:
        for i in range(lens):
            try:
                t =  float(line[i])
                if t >= max_list[i]:
                    max_list[i] = t
                if t <= min_list[i]:
                    min_list[i] = t
            except:
                pass
    
    for i in range(len(contents)):
        #print(contents[i])
        for j in range(lens):
            m = max_list[j] - min_list[j]
            if  m != 0 and min_list[j] >= 0:
                #contents[i][j] = float(contents[i][j])/max_list[j]
                contents[i][j] = (float(contents[i][j]) - min_list[j])/m
            elif m != 0  and min_list[j] < 0:
                if float(contents[i][j]) >= 0:
                    contents[i][j] = float(contents[i][j])/max_list[j]
                else:
                    contents[i][j] = 1 + float(contents[i][j])/min_list[j]
                
            else:
                contents[i][j] = ((max_list[j] + min_list[j])/2 - min_list[j])

    contents = gnn(contents, lists)  
    return (contents)


def read_file0(filename):

    #file_name = './data/cnews/Australian.csv'
    f = open(filename,'r',encoding='gbk')
    file_content=csv.reader(f)
    next(file_content)
    contents, labels = [], []
    for row in file_content:
        tmp_contents = []
        for i in range(1, len(row)):
            number = to_number(row[i]) 
            tmp_contents.append(float(number))
        contents.append(tmp_contents)
        labels.append(int(row[0]))
        #print(tmp_contents)
    #contents = data_normalization(contents)
    
    return contents, labels


def read_file1(filename):

    #file_name = './data/cnews/Australian.csv'
    f = open(filename,'r',encoding='gbk')
    file_content=csv.reader(f)
    next(file_content)
    contents, labels = [], []
    for row in file_content:
        tmp_contents = []
        for i in range(0, len(row)-1):
            number = to_number(row[i]) 
            tmp_contents.append(float(number))
        contents.append(tmp_contents)
        labels.append(int(row[-1]))
        #print(tmp_contents)
    contents = data_normalization(contents)
    
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):

    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data) 
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_category():

    categories = ['0','1']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def class_label(filename):
    for key in data_dict:
        if filename.find(key) != -1:
            return data_dict[key]
        
    return -1


def process_file(filename, cat_to_id,max_length=600):

    
    
    label = class_label(filename)
    if label == 0:
        contents, labels = read_file0(filename)
    else:
        contents, labels = read_file1(filename)

    x_pad = kr.preprocessing.sequence.pad_sequences(contents, max_length, dtype='float')
    y_pad = kr.utils.to_categorical(labels, num_classes=len(cat_to_id))
    return x_pad, y_pad


def batch_iter(x, y, batch_size=128):

    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


