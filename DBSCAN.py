#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
from sklearn.cluster import DBSCAN

def img_pre(img):
    img = img[ : , : , (2, 1, 0)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if((img[i][j][2]!=0) and (img[i][j][0]==0) and (img[i][j][1]==0)):
                img[i][j][2]=255
            if((img[i][j][0]!=0) and (img[i][j][1]==0) and (img[i][j][2]==0)):
                img[i][j][0]=255
            if((img[i][j][0]==img[i][j][1]) and (img[i][j][1]==img[i][j][2]) and (img[i][j][0]!=0)):
                img[i][j][0]=255;img[i][j][1]=255;img[i][j][2]=255      

            if(img[i][j][2]>img[i][j][1]):
                if(img[i][j][2]>img[i][j][0]):
                    img[i][j][1]=0;img[i][j][0]=0;img[i][j][2]=255
            if(img[i][j][0]>img[i][j][1]):
                if(img[i][j][0]>img[i][j][2]):
                    img[i][j][1]=0;img[i][j][2]=0;img[i][j][0]=255
    return img

def coor(imgp):
    coor=[]
    for i in range(imgp.shape[0]):
        for j in range(imgp.shape[1]):
            if((imgp[i][j][0]==255) and (imgp[i][j][1]==255) and (imgp[i][j][2]==255)):
                coor.append([j,i])
            if((imgp[i][j][0]==0) and (imgp[i][j][1]==0) and (imgp[i][j][2]==255)): #保留地表基線(藍線)
                coor.append([j,i])
    return(np.array(coor))



path = './all/'
count = 0
for root,dirs,files in os.walk(path):
      for each in files:
          if each.endswith('png'):
             count += 1

df = pd.read_csv("./liddar_img_v1.csv")

DList=[]
for i in range(df.shape[0]):
    file=df['image'][i]
    img0=cv2.imread(path+file)
    img0 = img0[ : , : , (2, 1, 0)]
    imgp=img_pre(img0)
    X=coor(imgp)
    y_pred = DBSCAN(eps = 64).fit_predict(X)
    result=len(set(y_pred))
    DList.append(result)
    print(i,file,result)

with open("DList.txt", "w") as output:
    output.write(str(DList))

