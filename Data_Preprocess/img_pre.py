from numpy.random import randint
import numpy as np
import cv2
from matplotlib import pyplot as plt

def img_pre(img):
    img = img[ : , : , (2, 1, 0)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # (0,0,B):(0,0,0)
            if((img[i][j][2]!=0) and (img[i][j][0]==0) and (img[i][j][1]==0)):
                img[i][j][2]=0
            # (R,0,0):(0,0,0)
            if((img[i][j][0]!=0) and (img[i][j][1]==0) and (img[i][j][2]==0)):
                img[i][j][0]=0
            # R=B=G: (255,255,255) & keep (0,0,0)
            if((img[i][j][0]==img[i][j][1]) and (img[i][j][1]==img[i][j][2]) and (img[i][j][0]!=0)):
                img[i][j][0]=255;img[i][j][1]=255;img[i][j][2]=255      
            # B>R & G:(0,0,0)
            if(img[i][j][2]>img[i][j][1]):
                if(img[i][j][2]>img[i][j][0]):
                    img[i][j][1]=0;img[i][j][0]=0;img[i][j][2]=0
            # R>G & B:(0,0,0)
            if(img[i][j][0]>img[i][j][1]):
                if(img[i][j][0]>img[i][j][2]):
                    img[i][j][1]=0;img[i][j][2]=0;img[i][j][0]=0
    return img

def img_compare(img_file,path):
    ix = randint(0, count, 1)
    file=img_file[ix[0]]
    img=cv2.imread(path+file)
    img = img[ : , : , (2, 1, 0)]
    img2=img_pre(img)
    plt.figure(figsize=(14,14))
    plt.subplot(1,2,1)
    plt.imshow(img);plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img2);plt.axis('off')
    print(file)