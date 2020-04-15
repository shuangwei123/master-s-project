#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3

import os
import sys
import csv
base_path = os.getcwd()
sys.path.append(base_path)

class FileProcess():
    
   
    def file_get_name(self, file_dir):
        file_list = []
        for root, dirs, files in os.walk(file_dir):
            for f in files:
                file_list.append(f)                

            return file_list
    
    
    def file_get_headers(self, file):
        f = open(file,'r',encoding='gbk')
        file_content=csv.reader(f)

        headers = []
        for line in file_content:
            headers = line
            break
            
        return headers


    def file_get_data(self, file, header):
        f = open(file,'r',encoding='gbk')
        file_content=csv.reader(f)
        data_list = []
        idx = 0

        headers = []
        for line in file_content:
            for i in range(len(line)):
                if line[i] == header:
                    idx = i
                    break
            break
            
        for line in file_content:
            if header in line:  
                continue
            data_list.append(line[idx])
            
        return data_list
            
        #return data_list            
            
    
    def file_get_data_txt(self, file):
        f = open(file, 'r')
        result_list = []
        for line in f:
            line = line.replace('\n','')
            tmp = line.split(' ')
            result_list.append(tmp)
        
        return result_list
    
    
    def write_csv(self, file, headers, data):
        with  open(file,'w') as csvFile:
            writer = csv.writer(csvFile)
            
            writer.writerow(headers)
            
            for i in range(len(data)):
                writer.writerow(data[i]) 
            
            
            
                    
            
        
        
#f = FileProcess()
#file_list = f.get_file_name('../hs300')
#f.get_date('../hs300' + '/' + file_list[0], 'close')
#print(file_list)
#print(file)

