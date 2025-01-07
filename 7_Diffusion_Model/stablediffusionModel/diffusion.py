import torch
import torch.nn as nn
from torch.nn import functional as Fn
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, n_embed):
        super().__init__()

        self.linear1 = nn.Linear(in_features=n_embed, out_features=4 * n_embed)
        self.linear2 = nn.Linear(in_features=4 * n_embed, out_features=4 * n_embed)

    def forward(self, x:torch.Tensor):
        '''
        Nothing fancy hear, just 2 linear layers with Silu Activation
        input is (1, 320) --> (1, 1280)
        '''

        x = self.linear1(x)

        x = nn.SiLU(x)

        x = self.linear2(x)

        return x


class Diffusion:
    '''
    It is basically our diffusion architecture which kcombines all the procedure
    into one of them

    '''

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = UnetOutputLayer(320, 4)

    def forward(self, latent:torch.Tensor, context: torch.Tensor, time:torch.Tensor):
        '''
        basically our model will have input from the image latents with encoded image
        the context or the word embedding of the current image
        the time embedding which will basically add noise in each time stamp

        latent (image) - > batch_size, 4, height / 8, width /8
        context(text) -> batch_size, seq_len, dim {batch_size, height*width, channel}
        time -> (1, 320) # a column vector containing time embeddings
        '''

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # batch_size, 4, height / 8, width /8 -> batch_size, 320, height / 8, width /8
        # so basically we will build an additional layer at the end to the UNet to match the total number of 
        # output of original one, but prior to that we output total 320 embedding 
        output = self.unet(latent, context, time)

        # batch_size, 320, height / 8, width /8 -> batch_size, 4, height / 8, width /8
        # Now our input and output dimension of Unet matches
        output = self.final(output)

        return output


