# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import segyio
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog
'''
train_data_mat = loadmat("../data/train_data2.mat")
 
train_data = train_data_mat["Data"]
train_label = train_data_mat["Label"]
 
print(train_data.shape, train_label.shape)
'''
print("JAM_PROCESSING")
print("请选择处理模块")
print("剩余静校正处理按1，噪声衰减按2")

jam = int(input())

if jam==1:
    print("剩余静校正处理需要动校拉伸并且切除过的cmp道集或者偏移并切除过的cip道集")
    print("剩余静校正处理需要从道集对应的剖面上拾取某一单一层位，层位尽量从剖面开始连续至结尾")
    print("所拾取的层位尽量保证上下一定范围内没有第二套波阻")
    print("所要求的层位文件为文本格式，内容分两列，第二列单位为毫秒或者米，例如")
    print("cmp      time")
    print("1001" "     " "2100")
    print("1002" "     " "2100")
    print("1003" "     " "2100")

    # --------------------读sgy为矩阵------------------------#
    # 初始化一个列表存数据，最后再转为矩阵
    
    #for tt in range(5998140,5999098):
    #name1=r"D:\line2000\cmp5998471.segy"
    name1=r"D:\line2000\cip_1078.sgy"
    root = tk.Tk()
    root.withdraw()
    name1 = filedialog.askopenfilename() #获得选择好的文件
    #获得选择好的文件
   
        
    #    name1=r"C:\Users\wts_duxin\Desktop\line2000\cmp"+str(tt)+".segy"
    data_list = []
        # sgy文件路径
    pp_path = name1
    with segyio.open(pp_path, mode="r+", iline=17,
                     xline=13,
                     strict=False,
                     ignore_geometry=False) as f:
            for i in range(len(f.trace)):
                data_list.append(f.trace[i])
                # 转为矩阵
    data_numpy = np.array(data_list)
       
        #exec(f'cmp{tt}=data_numpy')
    #train_data=data_numpy
    train_data=np.transpose(data_numpy)
    train_data1=train_data[550:750,:]
    train_data2=train_data[779:795,:]
    
    max_amp_local=np.zeros(train_data2.shape[1])
    statics=np.zeros(train_data2.shape[1])
    
    
    for tt in range(0,train_data2.shape[1]):
          
          max_amp_local[tt]=np.argmax(abs(train_data2[:,tt]))
          if max_amp_local[tt] ==0:
             statics[tt]=0
          elif max_amp_local[tt] !=0:
            statics[tt]=(max_amp_local[tt]-5)*(-10)
            
    cc=np.linspace(1,statics.shape[0],statics.shape[0])
    bbbb=np.vstack((cc,statics)) 
    
    output_statics=np.transpose(bbbb)
    np.savetxt(fname="d:/statics.txt",X=output_statics)
    print("  ")
    print("  ")
    print("  ")
    print("剩余静校正计算完成")
    print("END")

    
    
    '''
    f = open('d:/statics.txt','w')
    f.write(statics+'/n')
    f.close()
    '''
else:
        print("未进行静校正计算")


