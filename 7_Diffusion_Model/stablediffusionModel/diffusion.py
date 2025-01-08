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

class SwitchSequential(nn.Sequential):
    '''
    The image embedding which is latent
    THe sentence embedding which is Context
    The time embedding which is time

    UnetAttentionBlock so, how will you send image and Context combinely into UNET
    thats where Cross attention comes, here The UnetAttention block will calculate the cross 
    attention between the latents and Context and send them as input into Unet

    UnetResidualBlock will match our latents with the time step

    '''

    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UnetAttentionBlock):
                latent = layer(latent, context)
            elif isinstance(layer, UnetResidualBlock):
                latent = layer(latent, time)
            else:
                latent = layer(latent)
        
        return latent




class UNET(nn.Module):
    '''
    So Unet is made up of Encoder, Decoder and a bottle neck layer in between both of them
    There are also Skip Connections if you see in between the Encoders and Decoders which connects them 

    
    '''
    def __init__(self,):
        super().__init__()

        self.encoders = nn.Module([
            # bathc_size, 4, height/8, width/8 -> bathc_size, 640, height/8, width/8
            SwitchSequential(nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)),

            SwitchSequential(UnetResidualBlock(320, 320), UnetAttentionBlock(8, 40)),

            SwitchSequential(UnetResidualBlock(320, 320), UnetAttentionBlock(8, 40)),

            # bathc_size, 320, height/8, width/8 -> bathc_size, 320, height/16, width/16
            SwitchSequential(nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1, stride=2)),

            SwitchSequential(UnetResidualBlock(320, 640), UnetAttentionBlock(8, 80)),

            SwitchSequential(UnetResidualBlock(640, 640), UnetAttentionBlock(8, 80)),

            # bathc_size, 640, height/16, width/16 -> bathc_size, 640, height/32, width/32
            SwitchSequential(nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1, stride=2)),

            SwitchSequential(UnetResidualBlock(640, 1280), UnetAttentionBlock(8, 160)),

            SwitchSequential(UnetResidualBlock(1280, 1280), UnetAttentionBlock(8, 160)),

            # bathc_size, 1280, height/32, width/32 -> bathc_size, 1280, height/64, width/64
            SwitchSequential(nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, padding=1, stride=2)),

            SwitchSequential(UnetResidualBlock(1280, 1280)),

            # bathc_size, 1280, height/64, width/64
            SwitchSequential(UnetResidualBlock(1280, 1280))



        ])


class Diffusion:
    '''
    It is basically our diffusion architecture which kcombines all the procedure
    into one of them

    '''

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

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


