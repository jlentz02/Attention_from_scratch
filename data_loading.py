import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import csv



#Data loading
#Stolen from my own project JEPA
#Takes as argument the data size and the batch size
#Returns a pytorch DataLoader for training
def load_data(batch_size, data_size = None):
    with open("data/mnist_train.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train = list(csv_reader)
    train = train[1:]
    train = [[int(train[i][j]) for j in range(len(train[0]))] for i in range(len(train))]
    train = torch.tensor(train, dtype = torch.float32)

    #Data formatting and data loader
    train_Y = train[:,0].long()
    train_X = train[:,1:]

    #Trim data set down
    if data_size is not None:
        train_X = train_X[:data_size]
        train_Y = train_Y[:data_size]

    dataset = TensorDataset(train_X, train_Y)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last=True)

    return dataloader


