import torch
from torch import nn as nn
from torch.nn import functional as Fn
import math


class SelfAttention(nn.Module):
    '''
    - Self Attention Mechanism, here in heads are total number of heads that are split after multiplication of
        input matrix with Q, K, V
    - the embedding dimension (which mean feature vector across) is the number of channels
    - in_proj_bias is the bias term which we add in the input matrix, (inp matrix when multiplied 
        to K,Q,V seperately takes in effect )
    - out_proj_bias is the bias term which we add in the output matrix, (out matrix when multiplied 
        to each(Q, K, V) concatenated heads after mutilplitcation with INP matrix combinely takes in effect)

    '''

    def __init__(self, heads:int, d_embed:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(in_features=d_embed, out_features=3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(in_features=d_embed, out_features=d_embed, bias=out_proj_bias)
        self.n_heads = heads
        self.d_head = d_embed // heads

    def forward(self, x:torch.Tensor, mask = False):
        '''
        the input tensor will be split into Q, K, V matrix
        which then multiplied seperately with Input matrix
        the multiplied res will then concatenated together 
        the concatenated matrix will be given as Output after multiplying it with
        output matrix

        (x:batch_size, seq_len, Dim) which is (batch_size, height*width, channel) of an image
        '''

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # batch_size, seq_len, d_embed -> batch_size, seq_len, d_embed * 3 --> 3 tensors of batch_size, seq_len, d_embed
        query, key, value = self.in_proj(x).chunk(3, dim = -1)

        # batch_size, seq_len, d_embed -> (batch_size, seq_len, H, d_head / H) -> (batch_size, H, seq_len, d_head/H)
        query = query.view(intermediate_shape).transpose(1, 2)
        key = key.view(intermediate_shape).transpose(1, 2)
        value = value.view(intermediate_shape).transpose(1, 2)

        # batch, H, seq_len, seq_len 
        weight = query @ key.transpose(-1, -2)

        if mask:
            # upper triangular matrix all elements down diagonal will be 0
            maskMat = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(maskMat, -torch.inf)

        weight = weight / math.sqrt(self.d_head)

        weight = Fn.softmax(weight, dim=-1)
        
        # batch, H, seq_len, seq_len * batch, H, seq_len, dim/head  -> batch, head, seq_len, dim/head
        output = weight @ value

        #  batch, head, seq_len, dim/head ->  batch, seq_len, head, dim/head
        output = output.transpose(1, 2)

        # batch, seq_len, head, dim/head  -> batch, seq_len, dim
        output = output.reshape(input_shape)

        # multiply with out matrix
        output = self.out_proj(output)

        # batch, seq_len, dim
        return output






