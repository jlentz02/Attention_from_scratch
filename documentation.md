## Documentation of Experiments
# Basic Multi-head Transformer
    My first model in this project is a 2 block transformer model and a linear input and linear output layer. 
    Each transformer layer has a multi-head attention layer and a feed-forward layer that combine to make one block.

    For my first batch of experiments I am using the MNIST data set because of its simplicity and fast training times.
    To facilitate transforming the 28 x 28 image into tokens, I have prepended a "tokenizer" that transforms each image
    into a list of k x k patches where k = 28/patch_size. These tokens are then fed into the first linear linear of the model.
    I am training the model, not to predict the next token, but to predict the label of the image. This is facilitated by
    using a CLS token.

    So far, I have observed that this is probably not a great use case for a transformer model. I say this because using
    fewer tokens has resulted in better model performance. In particular, setting patch_size = 14 (resulting in 4 tokens)
    has resulted in the best predictive performance. However, you could argue this makes the most sense. If we chose 
    patch_size = 4 (yielding 49 tokens), what does each patch capture? A few pixels of a curve? Most of them will 
    simply see empty space. This limits the capability of the model to generalize patterns of pixels across disparate 
    images. 

    While it is not really the point of this project (that would be learning about attention, its variants, and use cases),
    I will still include some results of the models runs. 

    | model_dim | patch_size | heads | optimizer | lr   | batch_size | epochs | train accuracy | test accuracy |
    |-----------|------------|-------|-----------|------|------------|--------|----------------|---------------|
    | 64        | 14         | 8     | adam      | 1e-3 | 32         | 50     | 99.3%          | 96.8%         |

