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


class UnetAttentionBlock(nn.Module):
   '''
   UnetAttentionBlock so, how will you send image and Context combinely into UNET
    thats where Cross attention comes, here The UnetAttention block will calculate the cross 
    attention between the latents and Context and send them as input into Unet
   '''
   
   def __init__(self, n_heads:int, n_embed:int, d_context=768):
       super().__init__()
       channels = n_heads * n_embed

       self.groupNorm = nn.GroupNorm(32, num_channels=channels, eps=1e-6)
       self.conv_input = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
       self.layer_norm1 = nn.LayerNorm(channels)
       self.attention_1 = SelfAttention(heads=n_heads, d_embed=channels, in_proj_bias=False)
       self.layer_norm2 = nn.LayerNorm(channels)
       self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias = False)
       self.layer_norm3 = nn.LayerNorm(channels)
       self.linear1 = nn.Linear(in_features=channels, out_features=4 * channels * 2)
       self.linear2 = nn.Linear(in_features=4 * channels * 2, out_features= channels )
       self.conv_output = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

   def forward(self, x, context):
       #    batch_size, features, height, width
       # context: (batch_size, seq_len, Dim)
       residue_long = x

       x = self.groupNorm(x)
       x = self.conv_input(x)

       n, c, h, w = x.shape

      #  batch_size, features, height, width -> batch_size, features, height * width
       x= x.view((n, c, h * w))

       # batch_size, features, height * width -> batch_size, height * width, features
       x = x.transpose(-1. -2)

       # Normailization with Self attention and skip Connection


       residue_short = x

       x = self.layer_norm1(x)
       self.attention_1(x)
       x += residue_short

       # Normalization + cross Attention and skip conection

       residue_short = x

       x = self.layer_norm2(x) 
       self.attention_2(x) # cross attention
       x += residue_short


       # Normalization + Feed Forward +  Skip Connection
       residue_short = x

       x = self.layer_norm3(x)
       x, gate = self.linear1(x).chunk(2, dim = -1)

       x = x * Fn.gelu(gate)

       x = self.linear2(x)

       x += residue_short

       # batch_size, height * width, features -> batch_size, features, height * width
       x = x.transpose(-1. -2)

       #  batch_size, features, height * width -> batch_size, features, height, width
       x= x.view((n, c, h,  w))

       return self.conv_output(x) + residue_long

       





class UnetResidualBlock(nn.Module):

    '''
    Basically this block merges the TIme Embed, to the Inp Vector of Latents and context
    It is just to relate the time embedding to latents and contexts
    Hence the output will also be varied based on time at whihc Noise is infused
    '''

    def __init__(self, inChannel, outChannel, n_time = 1280):
        super().__init__()
        self.groupNorm1 = nn.GroupNorm(num_groups=32, num_channels=inChannel)
        self.conv1 = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, padding=1)
        self.linearTime = nn.Linear(in_features=n_time, out_features=outChannel)
        self.groupNorm2 = nn.GroupNorm(num_groups=32, num_channels=outChannel)
        self.conv2 = nn.Conv2d(in_channels=outChannel, out_channels=outChannel, kernel_size=3, padding=1)

        if inChannel == outChannel:
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=1, padding=0)


    def forward(self, feature, time):
        # batch_size, inChannel, height, width ; Time(1, 1280)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        residue = feature
        feature = self.groupNorm1(feature)
        feature = Fn.silu(feature)
        feature = self.conv1(feature)
        time = self.linearTime(time)

        # We are merging the time embedding and features, now as time embed is not same increasing it's size by unsqueezinf
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
    
        merged = self.groupNorm2(merged)
        merged = Fn.silu(merged)
        merged = self.conv2(merged)
        
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -->
        # --> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residualLayer(residue)






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




class Upsample(nn.Module):

    def __init__(self, channels:int ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        # batch_suze, channel, height, width -> batch_suze, channel, height * 2, width * 2
        x = Fn.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        return x
        




class UNET(nn.Module):
    '''
    So Unet is made up of Encoder, Decoder and a bottle neck layer in between both of them
    There are also Skip Connections if you see in between the Encoders and Decoders which connects them 

    
    '''
    def __init__(self,):
        super().__init__()

        # Increase the Number of Features and decrease it's size
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
        
        # the bottle Neck Layer all features are squezzed
        self.bottleNeck = SwitchSequential(
            UnetResidualBlock(1280, 1280), 
            UnetAttentionBlock(8, 160),
            UnetResidualBlock(1280, 1280)
        )

        # Do the opposite of Encoder, decrease the number of features and increase the image size
        self.decoder = nn.Module([
            # why is the input of Residual block doubled, now when you see the diagram
            # there is a skip connection from encoder to decoder, that skip 
            # connections adds up same amount of layer on top of it, so
            # the input will be double of it.

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UnetResidualBlock(2560, 1280)),

            SwitchSequential(UnetResidualBlock(2560, 1280)),

            SwitchSequential(UnetResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),

            SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),

            SwitchSequential(UnetResidualBlock(1920, 1280), UnetAttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UnetResidualBlock(1920, 640), UnetAttentionBlock(8, 80)),

            SwitchSequential(UnetResidualBlock(1280, 640), UnetAttentionBlock(8, 80)),

            SwitchSequential(UnetResidualBlock(960, 640), UnetAttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UnetResidualBlock(960, 320), UnetAttentionBlock(8, 40)),

            SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 40)),

        ])






class UNETOutputLayer(nn.Module):

    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=inChannel)
        self.conv = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, padding=1)

    def forward(self, x):
        # batch_size, 320, height / 8, width /8 -> batch_size, 4, height / 8, width /8
        x = self.groupnorm(x)
        x = Fn.silu(x)

        x = self.conv(x)

        # batch_size, 4, height / 8, width /8
        return x









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


