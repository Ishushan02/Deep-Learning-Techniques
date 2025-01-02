import torch
import torch.nn as nn
from torch.nn import functional as Fn
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VariationalAutoEncoder(nn.Sequential):

    '''
    So, basically the work of the Encoder is to reduce the dimension
    of the images whereas increasing the number of features in it.

    '''
    def __init__(self):
        super.__init__(
            # inp (batch, channel, height, width) -> (batch, 128, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # combination of Convolution and Normalization
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # basically we are decreasing the size of the image
            # (batch_size, 128, height, width) -> (batch_size, 128, height/ 2, width / 2) # out shape {ceil(n + 2p -f / s)  + 1}
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height /2, width / 2) --> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, height /2, width / 2) --> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height/4, width/4) --> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height/4, width/4) --> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) --> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) --> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) --> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, height/8, width/8) --> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # Normalization Layers (batch_size, 512, height/8, width/8)
            nn.GroupNorm(num_groups=32, num_channels=512),

            # sigmoid Linear Unit F(x) = x * sigmoid(x)
            nn.SiLU(),

            # (batch_size, 512, height/8, width / 8) --> (batch_size, 8, height/8, width / 8)
            # this is the bottleneck layer of the Encoder
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=0),

            # (batch_size, 8, height/8, width / 8) --> (batch_size, 8, height/8, width / 8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),

        )

    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        # x is (batch_size, channel, height, width)
        # nois is (batch, channel, height/8, width / 8)

        # we pass this input image into the self module we have definer
        for each_module in self:

            # wherever there is stride of 2, we are modifying to add the stride to the right and bottom corners
            if getattr(each_module, "stride") == (2, 2):
                # (paddingLeft, paddingRight, paddingTop, paddingBottom)
                x = Fn.pad(x, (0, 1, 0, 1))
            
            x = each_module(x)

        # as this is Variational AutoEncoder we learn the mean and the variance of the Gaussian distribution hence,
        # (batch, 8, height/8, width/8) -> (2 tensors seperated across channels) (batch, 4, height/8, width/8)
        mean, logVariance = torch.chunk(input=x, chunks=2, dim=1)

        # clamping means making the output in range of -30 to 20, such that too small or too large value doesn't come
        logVariance = torch.clamp(logVariance, min=-30, max=20)

        # getting out variance of logVariance
        variance = logVariance.exp()

        # standard deviation
        std = variance.sqrt()

        # So, now basically we have mean and variance of the distribution,
        # but we want our distribution to be normalized with mean = 0 and std = I
        # so, Z(0, 1) -> X(mean, std)
        # How do we convert it , X = mean + std * Z (this is called as Sampling from distribution)
        X = mean + std * noise

        # Scaling it for better calculation (It is there in Original Paper)
        X *= 0.18215

        return X

