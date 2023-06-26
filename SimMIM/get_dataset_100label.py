import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def rand_bbox(size,mask_ratio):
    length = size[2]
    cut_length = np.int(length*mask_ratio)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def MaskData(data, mask_ratio):
    bbx1, bbx2 = rand_bbox(data.size(), mask_ratio)
    data[:, :, bbx1: bbx2] = torch.zeros((data.size()[1],bbx2-bbx1)).cuda()
    return  bbx1, bbx2, data

def TrainDataset(num, rand_num):
    x = np.load(f"/data/fuxue/Dataset_4800/X_train_{num}Class.npy")
    y = np.load(f"/data/fuxue/Dataset_4800/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train_labeled, X_val, Y_train_labeled, Y_val = train_test_split(x, y, test_size=0.3, random_state=rand_num)
    return X_train_labeled, X_train_labeled, X_train_labeled, X_val, Y_train_labeled, Y_train_labeled, Y_train_labeled, Y_val

def TestDataset(num):
    x = np.load(f"/data/fuxue/Dataset_4800/X_test_{num}Class.npy")
    y = np.load(f"/data/fuxue/Dataset_4800/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y

if __name__ == '__main__':
    X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val = TrainDataset(10)