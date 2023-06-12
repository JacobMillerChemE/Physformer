import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class SelfAttention(nn.Module):
    """Accepts a tokenized, batched input tensor (x) and returns Attention(x).

    The given input tensor, x, must be of shape (B, S, embedding dim). Where
    B is batch size and S is context length.This class will then take that input 
    tensor and perform multihead attention (masked if provided). The class will 
    return a Tensor of the shape: (B, S, embedding dim).

    ...
    Attributes
    ----------
    mask : Tensor
        The mask provided to make the Transformer unidirectional
    NH : int
        The number of attention heads
    d_m : int
        The length of the embedding dimension 
    d_k : int
        The embedding dimension for a single attention head found by NH/d_m
    Q : nn.Linear
        The query matrix as produced by multiplying x by a set of weights, Wq
    K : nn.Linear
        The key matrix as produced by multiplying x by a set of weights, Wk
    V : nn.Linear
        The value matrix as produced by multiplying x by a set of weights, Wv
    batch_size : int
        the batch size during training
    context_length : int
        The number of items in the input sequence (e.g. # words in a sentence)
    
    Methods
    -------
    head_splitter(x, qkv):
        Reshapes QKV from (B, S, d_m) to (B, S, NH, d_k)

    """

    def __init__(self, NH: int, embed_dim: int, mask=None):
        """

        Args:
        -----------
        mask : Tensor
            The mask provided to make the Transformer unidirectional
        NH : int
            The number of attention heads
        d_m : int
            The length of the embedding dimension 
        d_k : int
            The embedding dimension for a single attention head found by NH/d_m
        Q : nn.Linear
            The query matrix as produced by multiplying x by a set of weights,Wq
        K : nn.Linear
            The key matrix as produced by multiplying x by a set of weights, Wk
        V : nn.Linear
            The value matrix as produced by multiplying x by a set of weights,Wv
        W_0:
            Linear layer for after concatentation has been performed
        """

        super().__init__()
        self.mask = mask
        self.NH = NH
        self.d_m = embed_dim
        self.d_k = embed_dim // self.NH
        assert self.d_m % self.NH == 0, "Model dimension not divisible by heads"

        # Create Query, Key, and Value Matrices in bulk
        self.Q = nn.Linear(self.d_m, self.d_m, bias=False)
        self.K = nn.Linear(self.d_m, self.d_m, bias=False)
        self.V = nn.Linear(self.d_m, self.d_m, bias=False)

        # Linear layer for after concatentation
        self.W_0 = nn.Linear(self.d_m, self.d_m, bias=False)


    def head_splitter(self, x: Tensor, qkv: list):
        """Takes in QKV matrices and reshapes them to (B, S, NH, d_k).

        ...
        Args:
        -----------
        x : Tensor
            Input tensor into the transformer of size (B, S, d_m).
        qkv : list
            List containing Q, K, V matrices of shape (B, S, d_m)
        
        Returns:
        --------
        QKV : list
            List containing Q, K, V matrices of shape (B, S, NH, d_k)
        """

        self.batch_size = x.shape[0]
        self.context_length = x.shape[1]
        new_shape = (self.batch_size, self.context_length, self.NH, self.d_k)

        QKV = []
        for matrix in qkv:
            matrix = torch.reshape(matrix, new_shape)
            QKV.append(matrix)

        return QKV


    def forward(self, x: Tensor) -> Tensor:
        qkv = [self.Q(x), self.K(x), self.V(x)]

        query, key, value = self.head_splitter(x, qkv)
    
        # MatMul for Q and K.transpose but with einsum and scale by sqrt(d_k)
        QK_tran = torch.einsum("bqhd, bkhd -> bhqk", query, key)
        QK_tran = QK_tran / np.sqrt(self.d_k)

        if self.mask is not None:
            QK_tran = QK_tran.masked_fill(self.mask==0, float('-inf'))


        # Soft max to generate scaling values 
        QK_tran = torch.softmax(QK_tran, dim=2)

        # Matrix multiplication of KQ_tran and Value matrices 
        attention_heads = torch.einsum("bhqk, bkhd -> bhqd", QK_tran, value)

        # Concatenate heads to produce Tensor of shape (B, S, d_m)
        output_shape = (self.batch_size, self.context_length, self.d_m)
        attention = torch.reshape(attention_heads, output_shape)
        attention = self.W_0(attention)

        return attention


if __name__ == '__main__':
    X = torch.rand(10, 3, 6)
    attention = SelfAttention(3, 6)
    attn = attention.forward(X)
    print(attn[0, 0, :])
    print(attn.shape)
    

