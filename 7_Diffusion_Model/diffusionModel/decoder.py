import torch
import torch.nn as nn
from torch.nn import functional as Fn
from encoder import VariationalAutoEncoder
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    '''
    The process of Skip Connection such that the gradient should propagate
    We build it using groupNorm and Conv layer
    '''

    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.groupNorm1 = nn.GroupNorm(num_groups=32, num_channels=inChannels)
        # The height and width will remain same of Inp and Out Image {(ceil(n + 2p -f/ s)  + 1)}
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1)

        self.groupNorm2 = nn.GroupNorm(num_groups=32, num_channels=outChannels)
        # The dimension of image will remain same
        self.conv2 = nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1)

        '''
        Now defining the residual layer(the skip connection layer between the initial and final layer)
        so, let's say the inChannel and outChannel are same we can directly add them
        but if they are not same, we introduce a convNet b/w them and make there size same and then add them
        '''

        if(inChannels == outChannels):
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1, padding=0)
            # see kernel size so each output pixel is in correspondance of each input pixel, hence size will be same

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        We can also create Residual Block via this architecture 
            Conv->groupNorm->relu --> Conv->groupNorm->relu --> x + F(x)
        or the below one (both are correct groupNorm can be added initially or in back)
            groupNorm->silu->Conv --> groupNorm->silu->Conv --> x + F(x)
        '''
        # we get image with batch, inChannel, height, width    

        # storing initial layer for propagating it at the end (skip Connection and transfering initial Gradients)
        initialLayer = x # residue to pass to future layer
        x = self.groupNorm1(x)
        x = Fn.silu(x)
        x = self.conv1(x)

        x = self.groupNorm2(x)
        x = Fn.silu(x)
        x = self.conv2(x)

        return x + self.residualLayer(initialLayer) # skip connection and mitigating gradients
    
        
class VAE_AttentionBlock(nn.Module):

    def __init__(self, inChannel:int):
        super().__init__()
        self.groupNorm = nn.GroupNorm(num_groups=32, num_channels=inChannel)
        self.attention = SelfAttention(1, inChannel)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        '''
        Store initial layer for skip connection
        perfrom self attention (within each pixels of the feature vector, )
        return the output by adding residue to the self attention output
        '''

        initial_layer = x # residue to pass to future layer
        x = self.groupNorm(x)
        # (batch, channel, height, width)
        n, c, h, w = x.shape 

        # doing all the process for self attention (as Q, K and V are same in self attention)
        # (batch, channel, height, width) ->  (batch, channel, height * width)
        x = x.view(n, c, h * w)

        # (batch, channel, height * width) -> (batch, height * width, channel)
        x = x.transpose(-1, -2)

        # self attention so all Q, K, V will be same
        x = self.attention(x)

        # (batch, height * width, channel) -> (batch, channel, height * width)
        x = x.transpose(-1, -2)

        # (batch, channel, height * width) ->(batch, channel, height,  width)
        x = x.view((n, c, h, w))

        
        return x + initial_layer






class VariationalAutoDecoder(nn.Sequential):
    '''
    The main functionality of Decoder is to enhance the image dimension
    from the (reduced dimension) by Encoder
    '''

    def __init__(self):
        super.__init__(

        )

