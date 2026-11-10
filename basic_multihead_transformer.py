#Basic transformer with multihead attention

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import csv

torch.manual_seed(42)

#I have an 9070XT. Great gpu, but not for this
device = "cpu"
    
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, model_dim, key_dim, value_dim, heads):
        super().__init__()

        #Need model_dim/heads to be an integer
        assert value_dim%heads == 0, "Number of heads does not divide value dimension"
        assert key_dim%heads == 0, "Number of heads does not divide key dimension"

        head_key_dim = key_dim//heads
        head_value_dim = value_dim//heads

        self.head_key_dim = head_key_dim
        self.W_Q = nn.Parameter(torch.rand([heads, model_dim, head_key_dim]))
        self.W_K = nn.Parameter(torch.rand([heads, model_dim, head_key_dim]))
        self.W_V = nn.Parameter(torch.rand([heads, model_dim, head_value_dim]))
        self.W_0 = nn.Parameter(torch.rand([value_dim, model_dim]))

    def forward(self, x):
        Q = x@self.W_Q
        K = x@self.W_K
        V = x@self.W_V

        scale = self.head_key_dim**0.5
        scores = (Q@K.transpose(-2,-1))/scale
        weights = torch.softmax(scores, dim = -1)
        
        context = weights@V

        #Concatenation step
        context = context.transpose(0,1)
        context = context.reshape(context.shape[0], -1)

        return context@self.W_0

class MultiheadTransformerBlock(nn.Module):
    def __init__(self, model_dim, key_dim, value_dim, heads):
        super().__init__()

        #Attention
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = MultiheadAttentionLayer(model_dim, key_dim, value_dim, heads)

        #Feed Forward Network
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class MultiheadAttentionNN(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, key_dim, value_dim, heads):
        super().__init__()

        self.attention_linear = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            MultiheadTransformerBlock(model_dim, key_dim, value_dim, heads),
            MultiheadTransformerBlock(model_dim, key_dim, value_dim, heads),
            nn.Linear(model_dim, output_dim)
        )


'''
#Data loading
#Stolen from my own project JEPA
#Added way to trim data set
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
train_X = train_X[:1000]
train_Y = train_Y[:1000]

dataset = TensorDataset(train_X, train_Y)

batch_size = 64
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
'''

#Model stuff
model = MultiheadAttentionLayer(
    model_dim=32,
    key_dim=32,
    value_dim=32,
    heads=4
)
x = torch.rand((15,32))

y = model(x)

predictions = torch.argmax(y, dim =-1)

print(predictions)



