import torch 
import torch.nn as nn
from packages.utils.MHAttention import SelfAttention
import numpy as np

class DecoderLayer(nn.Module):
    """Accepts input tensor (x), performs 1 DecoderLayer pass & returns tensor.

    The given input tensor, x, must be of shape (B, S, embedding dim). Where
    B is batch size and S is context length. This class will then take that 
    input tensor and perform a pass through one Decoder Layer as follows: 
    Multihead self attention -> residual connection -> Layer normalization 
    -> Fully connected layer -> Skip connection -> Layer normalization. 


    ...
    Attributes
    ----------
    mask : Tensor
        The mask provided to make the Transformer unidirectional
    NH : int
        The number of attention heads
    d_model : int
        The length of the embedding dimension 
    LayerNorm : nn.LayerNorm
        Normalizes the input over the context length and embedding dimensions
    Attention : class
        Performs multihead self attention on given input
    FC : nn.Linear
        A two layer fully connected neural network with ReLU activation
    T : int
        The number of items in the input sequence (e.g. # words in a sentence)
    """
        
    def __init__(self, context_length, NH, d_model, ffc=256, mask=None):
        """Initializes a single decoder layer
        Attributes
        ----------
        mask : Tensor
            The mask provided to make the Transformer unidirectional
        NH : int
            The number of attention heads
        d_model : int
            The length of the embedding dimension 
        LayerNorm : nn.LayerNorm
            Normalizes the input over the context length & embedding dimensions
        Attention : class
            Performs multihead self attention on given input
        FC : nn.Linear
            A two layer fully connected neural network with ReLU activation
        T : int
            The number of items in the input sequence 
        ffc : int
            Number of hidden neurons in fully connected layer
        """

        super().__init__()
        self.T = context_length
        self.NH = NH
        self.d_model = d_model
        self.mask = mask
        self.LayerNorm1 = nn.LayerNorm((self.T, self.d_model))
        self.LayerNorm2 = nn.LayerNorm((self.T, self.d_model))
        self.Attention = SelfAttention(self.NH, self.d_model, self.mask)
        self.FC = nn.Sequential(
            nn.Linear(d_model, ffc),
            nn.ReLU(),
            nn.Linear(ffc, self.d_model)
        )

    
    def forward(self, x):
        out = self.Attention(x)
        out = out + x
        out = self.LayerNorm1(x)
        out_fc = self.FC(out)
        out = out_fc + out
        out = self.LayerNorm2(out)
        return out


class Transformer(nn.Module):
    """Accepts input tensor (x), stacks decoder layers & returns tensor.

    The given input tensor, x, must be of shape (B, S, embedding dim). Where
    B is batch size and S is context length. This class will perform positional
    embedding on the input tensor using the procedure from "Attention is All 
    you Need" and embed it into a desired latent dimension, d_model. The 
    Transformer will then pass the embedded representation through a stack of 
    Decoder Layers and pass the output from the stack to a final linear layer
    which will reduced the dimension from d_model to the original number of 
    input features, C.


    ...
    Attributes
    ----------
    mask : Tensor
        The mask provided to make the Transformer unidirectional
    NH : int
        The number of attention heads
    d_model : int
        The length of the embedding dimension 
    LayerNorm : nn.LayerNorm
        Normalizes the input over the context length and embedding dimensions
    Attention : class
        Performs multihead self attention on given input
    FC : nn.Linear
        A two layer fully connected neural network with ReLU activation
    T : int
        The number of items in the input sequence (e.g. # words in a sentence)
    channels : int
        Number of input features in the input array (e.g. Lorenz = 3 (x, y, z))
    heads : int
        Number of attention heads for mulithead self attention
    num_layers : int
        Number of decoder layers to stack 
    decoder_layer : class DecoderLayer()
        A single DecoderLayer() object for defining desired architecture
    Decoder : nn.ModuleList()
        A stack of DecoderLayers where number of layers == num_layers
    final_linear : nn.Linear()
        If d_k != d_model, can be used to project tensor size back to d_model
    embed : nn.Linear()
        Projects input tensor from given number of features to d_model
    invembed : nn.Linear()
        Projects output from last decoderLayer to orginal number of features


    ...
    Methods
    -------
    positional_encoding(x):
        Adds positional information to input tensor x as a sum of sines/cosines
    """
    def __init__(self,
                 context_length,
                 channels,
                 num_layers, 
                 d_model, 
                 heads,
                 ffc=256, 
                 mask=None):
        """
        Attributes
        ----------
        mask : Tensor
            The mask provided to make the Transformer unidirectional
        NH : int
            The number of attention heads
        d_model : int
            The length of the embedding dimension 
        LayerNorm : nn.LayerNorm
            Normalizes the input over the context length & embedding dimensions
        Attention : class
            Performs multihead self attention on given input
        FC : nn.Linear
            A two layer fully connected neural network with ReLU activation
        T : int
            The # of items in the input sequence (e.g. # words in a sentence)
        channels : int
            # of input features in the input array(e.g. Lorenz = 3 (x, y, z))
        heads : int
            Number of attention heads for mulithead self attention
        num_layers : int
            Number of decoder layers to stack 
        decoder_layer : class DecoderLayer()
            A single DecoderLayer() object for defining desired architecture
        Decoder : nn.ModuleList()
            A stack of DecoderLayers where number of layers == num_layers
        embed : nn.Linear()
            Projects input tensor from given number of features to d_model
        invembed : nn.Linear()
            Projects output from last decoderLayer to orginal number of features
        """

        super().__init__()
        self.mask=mask
        self.C = channels
        self.T = context_length
        self.d_model = d_model
        self.heads = heads
        self.num_layers = num_layers
        self.ffc = ffc
        self.decoder_params = (self.T,
                               self.heads, 
                               self.d_model,
                               self.ffc, 
                               self.mask)

        self.Decoder = nn.ModuleList([DecoderLayer(*self.decoder_params) 
                                      for i in range(self.num_layers)])
        self.embed = nn.Linear(self.C, self.d_model)
        self.invembed = nn.Linear(self.d_model, self.C)
    
    def postional_encoding(self, x):
        """
            Accepts input tensor of size B, S, d_model and adds positional info
            Returns tensor same size as input. 
        """
        
        pos = torch.arange(self.T)[:, None]
        i = torch.arange(self.d_model)[None, :]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(self.d_model))
        pe = pos * angle_rates
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        
        return pe+x
        
    def forward(self, x):
        out = self.embed(x)
        out = self.postional_encoding(out)
        for i in range(self.num_layers):
            out = self.Decoder[i](out)
        out = self.invembed(out)
        return out

    



if __name__ == "__main__":
    X = torch.rand(10, 3, 32)  # B x S x d_m
    norm_shape = X.shape[1:]
    physformer = Transformer(norm_shape)
    physformer_pass = physformer(X)
    print(physformer_pass)
    params = sum([p.numel() for p in physformer.parameters() if p.requires_grad])
    print(params)
 