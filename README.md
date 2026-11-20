# Hello
## Overview
This project, like most of my projects, is for my own learning. With this project, my aim was to better understand how the neural network architecture underpinning large language models (LLMs) functions. I refer to, of course, the transformer, first described in the seminal paper *Attention Is All You Need*. However, beyond just implementing the transformer architecture and experimenting with it on various datasets, I also intend to experiment with variants of the architecture. In particular, I am interested in things like sparse attention, pruning, and distillation. This list will be amended as the direction of inquiry evolves and as I read more papers. List below. 

## Summary of Files
* README.md: You're reading it. Thanks. 
* basic_multihead_transformer.py: This file contains the PyTorch code for a basic multi-head transformer with linear input, two transformer blocks, and linear output layer.
* basic_transformer.py: Similar to above, but without multi-head attention and without the tokenizer for the MNIST dataset.
* data_loading.py: Contains a single helper function for loading the MNIST dataset and returning a dataloader object.
* train_template.py: Contains all of the parts necessary to train the model including instantiation, data loading, optimizer, and the core training/evaluation loop. I wrote this file first as a .py file because it was easier to organize this way.
* training.ipynb: Contains essentially the same code as the train_template.py file, but split up into code blocks to facilitate easier repeat experimentation.   

## Papers I read for this project/psuedo-bibliography
* *Attention Is All You Need*, Vaswani et al
* *Kronecker Delta Rule for Long Context Linear Attention*, Hartl et al
* *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, Katharopoulos et al
