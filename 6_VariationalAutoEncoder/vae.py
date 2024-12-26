import torch
import torch.nn.functional as Fn
import torch.nn as nn


# Main Procedure is Input Image --> Hidden Dimension --> mean, variance, --> Parametarization trick, --> Decoder --> Output Image
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim = 100, z_dim = 20):
        super().__init__()

        #encoder
        self.inpimg_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_mean = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_var = nn.Linear(hidden_dim, z_dim)

        #decoder
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, input_dim)

        # Activations
        self.relu = nn.ReLU()

    def encoder(self, x):
        # qPhi(z | x)
        hidden_out = self.relu(self.inpimg_to_hidden(x))
        mean = self.hidden_to_mean(hidden_out)
        sigma = self.hidden_to_var(hidden_out)
        return mean, sigma


    def decoder(self, z):
        # ptheta(x|z)
        z_out = self.relu(self.z_to_hidden(z))
        return torch.sigmoid(self.hidden_to_out(z_out))

    def forward(self, x):
        mean, sigma = self.encoder(x)
        epsilon = torch.rand_like(sigma)
        z_parameterized = mean + sigma * epsilon
        img_constructed = self.decoder(z_parameterized)

        return mean, sigma, img_constructed

# test
# x = torch.randn(4, 28 * 28)
# vae = VariationalAutoEncoder(28 * 28)
# mean, sigma, outImg= vae(x)
# print(mean.shape, sigma.shape, outImg.shape)

if __name__=="__main__":
    
    pass
