import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import csv



#Data loading
#Stolen from my own project JEPA
#Takes as argument the data size, batch size, and the train test split (tts) ratio
#Returns a pytorch DataLoader for training
def load_data(batch_size, data_size = 60000, tts_ratio = 0.8):
    with open("data/mnist_train.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train = list(csv_reader)
    train = train[1:]
    train = [[int(train[i][j]) for j in range(len(train[0]))] for i in range(len(train))]
    train = torch.tensor(train, dtype = torch.float32)

    #Data formatting and data loader
    X = train[:,1:]
    Y = train[:,0].long()

    #Trim data set down
    train_size = int(data_size*tts_ratio)
    train_X = X[:train_size]
    train_Y = Y[:train_size]

    test_X = X[train_size:data_size]
    test_Y = Y[train_size:data_size]


    dataset_train = TensorDataset(train_X, train_Y)
    dataset_test = TensorDataset(test_X, test_Y)


    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = True, drop_last=True)

    return dataloader_train, dataloader_test

