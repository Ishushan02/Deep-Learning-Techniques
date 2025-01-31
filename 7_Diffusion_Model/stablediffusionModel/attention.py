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

        heads = head, d_embed = height * width
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
        # batchsize, height*width, channel
        batch_size, seq_len, d_embed = input_shape

        # batch_size, height * width, heads,  (batch_size, height*width, head, channel/heads)
        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # batch_size, seq_len, d_embed -> batch_size, seq_len, d_embed * 3 --> 3 tensors of batch_size, seq_len, d_embed
        # same as above --> OP is 3 tensors of (batch_size, height*width, channel)
        query, key, value = self.in_proj(x).chunk(3, dim = -1)

        # batch_size, seq_len, d_embed -> (batch_size, seq_len, H, d_head / H) -> (batch_size, H, seq_len, d_head/H)
        # same as above (batch_size, height*width, channel) --> (batch_size, height*width, head, channel/head) --> below
        #  --> batch_size, head, height*width, channel/head
        query = query.view(intermediate_shape).transpose(1, 2)
        key = key.view(intermediate_shape).transpose(1, 2)
        value = value.view(intermediate_shape).transpose(1, 2)

        # batch, H, seq_len, seq_len 
        # (batch_size, head, height*width, channel/head) * (batch_size, head, channel.head, head * width) --> below
        #  batch_size, head,  height*width, height*width
        weight = query @ key.transpose(-1, -2)

        if mask:
            # upper triangular matrix all elements down diagonal will be 0
            maskMat = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(maskMat, -torch.inf)

        weight = weight / math.sqrt(self.d_head)

        weight = Fn.softmax(weight, dim=-1)
        
        # batch, H, seq_len, seq_len * batch, H, seq_len, dim/head  -> batch, head, seq_len, dim/head
        # (batch_size, head,  height*width, height*width) . (batch_size, head,  height*width, channel/head) --> below
        # --> (batch_size, head, height*width, channel/head) 
        output = weight @ value

        #  batch, head, seq_len, dim/head ->  batch, seq_len, head, dim/head
        # (batch_size, head, height*width, channel/head)  --> (batch_size, height*width, head, channel/head) 
        output = output.transpose(1, 2)

        # batch, seq_len, head, dim/head  -> batch, seq_len, dim
        # (batch_size, height*width, head, channel/head) --> (batchsize, height*width, channel)
        output = output.reshape(input_shape)

        # multiply with out matrix
        # (batchsize, height*width, channel) * (batch, channel, channel)
        output = self.out_proj(output)

        # batch, seq_len, dim
        # (batch_size, head*width, channel)
        return output






class CrossAttention(nn.Module):
    '''
        n_head - total number of heads
        d_embed - embed dimension
        d_cross - dimension of keys and values
    '''
    def __init__(self, n_heads:int, d_embed:int, d_cross:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.q_proj = nn.Linear(in_features=d_embed, out_features=d_embed, in_proj_bias = in_proj_bias)
        self.k_proj = nn.Linear(in_features=d_embed, out_features=d_embed, in_proj_bias = in_proj_bias)
        self.v_proj = nn.Linear(in_features=d_embed, out_features=d_embed, in_proj_bias = in_proj_bias)

        self.out_proj = nn.Linear(in_features=d_embed, out_features=d_embed, out_proj_bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//self.n_heads

    def forward(self, x, y):
        '''
        x (latent) : batch_size, seq_len_Q, Dim_Q #Query
        y (context) : batch_size, seq_len_KV, Dim_KV #Key and Value
        '''

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Q = K * V 

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        query = q.view(interim_shape).transpose(1, 2)
        key = k.view(interim_shape).transpose(1, 2)
        value = v.view(interim_shape).transpose(1, 2)

        weight = query @ key.transpose(-1, -2)

        weight = weight / math.sqrt(self.d_head)

        weight = Fn.softmax(weight, dim=-1)

        output = weight @ value
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output





