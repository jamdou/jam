import numpy as np

aa=[[1,2,3],[4,5,6],[7,8,9]]
b=np.array(aa)


'''
for a in range(1,10):
     file1='d:/m/text'+str(a)+'.txt'
     fp=open(file1,'a+')
     print(aa,file=fp)
     fp.close()
'''



np.savetxt('d:/m/aa.csv', b, delimiter = ',')