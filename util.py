import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def prepareDataset():
    train=pd.read_csv("data/dataset.csv")
    train=train.set_index('image_name')
    X=[]
    for index,row in train.iterrows():
        img=cv2.resize(cv2.imread("data/images/"+index),(120,90))
        X.append(img)
    X=np.array(X)
    np.save("data/X.npy",X)
    Y=train.as_matrix()
    np.save("data/Y.npy",Y)

def load_dataset():
    X=np.load("data/X.npy")/255
    Y=np.load("data/Y.npy")
    m=X.shape[0]
    permutation=np.random.permutation(range(m))
    X=X[permutation]
    Y=Y[permutation]
    # 9 to 1 split between training and validation
    train_X=X[:int(0.9*m)]
    train_Y=Y[:int(0.9*m)]
    test_X=X[int(0.9*m):]
    test_Y=Y[int(0.9*m):]
    return train_X,train_Y,test_X,test_Y

#Intersection over Union- Accuracy Metrics
def iou(y,yh):
  x1=np.maximum(y[:,0],yh[:,0])
  x2=np.minimum(y[:,1],yh[:,1])
  y1=np.maximum(y[:,2],yh[:,2])
  y2=np.minimum(y[:,3],yh[:,3])
  i=np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
  u=(y[:,1]-y[:,0])*(y[:,3]-y[:,2])+(yh[:,1]-yh[:,0])*(yh[:,3]-yh[:,2])-i
  iou=i/u
  return np.sum(iou)/len(iou)
