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
        # x = [batch_size, tokens, model_dim]
        #self.W_Q = [heads, model_dim, head_key_dim]
        #TODO figure out how to make this work
        #want Q = [batch_size, heads, tokens, head_key_dim]
        #Using einsum: b, h, t, k
        Q = torch.einsum('btm,hmk -> bhtk', x, self.W_Q)
        K = torch.einsum('btm,hmk -> bhtk', x, self.W_K)
        V = torch.einsum('btm,hmk -> bhtk', x, self.W_V)

        scale = self.head_key_dim**0.5
        #Q@K.T = [batch_size, heads, tokens, model_dim]
        scores = (Q@K.transpose(-2,-1))/scale
        weights = torch.softmax(scores, dim = -1)
        
        #weights@V = [batch_size, heads, tokens, head_value_dim]
        context = weights@V

        #Concatenation step
        #Want to switch to [batch_size, tokens, heads, head_value_dim]
        context = context.transpose(1,2)
        #Concat to [batch_size, tokens, heads*head_value_dim]
        #Recall value_dim = heads*head_value_dim
        context = context.reshape(context.shape[0], context.shape[1], -1)
        
        #W_0 = [value_dim, model_dim]
        #output is [batch_size, tokens, model_dim] as desired
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
    
    
#Adding CLS token to this
class Tokenizer(nn.Module):
    def __init__(self, batch_size, patch_size, embedding_dim):
        super().__init__()

        self.embedding_layer = nn.Parameter(torch.rand([patch_size*patch_size, embedding_dim]))
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    def forward(self, x):
        #Only works for square images. 
        #Fine since we are starting with mnist
        x_side_len = int(x.shape[-1]**0.5)
        x = x.reshape(self.batch_size, x_side_len, x_side_len)
        grid_len = x_side_len//self.patch_size
        x = x.reshape(self.batch_size, grid_len, self.patch_size, grid_len, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4)

        patches = x.reshape(self.batch_size, grid_len**2, self.patch_size**2)
        
        tokens = patches@self.embedding_layer
        
        #We take the generated patches from above and append a CLS token 
        #that will be used for prediction
        #We might need to wrap this in with torch no_grad
        CLS_token = torch.rand([x.shape[0], 1, self.embedding_dim])

        x = torch.cat([CLS_token, tokens], dim=1)
        return x
        

class MultiheadAttentionNN(nn.Module):
    def __init__(self, batch_size, patch_size, output_dim, model_dim, key_dim, value_dim, heads):
        super().__init__()

        self.attention_linear = nn.Sequential(
            Tokenizer(batch_size, patch_size, model_dim),
            MultiheadTransformerBlock(model_dim, key_dim, value_dim, heads),
            MultiheadTransformerBlock(model_dim, key_dim, value_dim, heads),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        return self.attention_linear(x)
        
