import torch
import torchvision.datasets as datasets
from tqdm import tqdm # For progress bar
from torchvision import transforms
import torch.nn.functional as Fn
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# Main Procedure is Input Image --> Hidden Dimension --> mean, variance, --> Parametarization trick, --> Decoder --> Output Image
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim = 200, z_dim = 20):
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


device = torch.device("cpu")
input_size = 28 * 28
hiden_dimension = 200
z_dimension = 20
Num_Epochs = 20
batch_size = 128
learning_rate = 0.0001 #(3e-4 Karapathy constant, he uses mostly this value in his experiments)

# Dataset
dataset = datasets.MNIST("dataset", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
vae_model = VariationalAutoEncoder(input_dim=input_size, hidden_dim=hiden_dimension, z_dim=z_dimension).to(device)
optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
lossFn = nn.BCELoss(reduction="sum") # Binary Cross Entropy mostly used for 
                    # reduction = "mean"


# Start Training
for epoch in range(Num_Epochs):
    loop = tqdm(enumerate(train_loader))
    for i, (x, y) in loop:
        # FOrward pass
        x = x.to(device).view(x.shape[0], input_size)
        mean, sigma, outputImage = vae_model(x)

        #loss
        reconstructLoss = lossFn(outputImage, x) # reconstruction loss
        kl_divergenceLoss = -1 * torch.sum(1 + torch.log(sigma.pow(2)) - mean.pow(2)-sigma.pow(2)) #klDiv loss
        
        # backporop
        loss = reconstructLoss + kl_divergenceLoss
        optimizer.zero_grad() # no accumulated grad from before
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())



torch.save(vae_model.state_dict(), f = "/Users/ishananand/Desktop/Pytorch/Model/vae.pth")
print(f"Model saved to Model Path")

def inference(digit, num_examples = 1):
    '''
    Generate num_examples of a particular digit, we 
    generally extract an example of each digit and then we sample from mean and sigma 
    from its sample

    After sampling we can directly execute Decoder Block for generation
    '''

    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            print(images[d].shape)
            mu, sigma = vae_model.encoder(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = vae_model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        # out.reshape(128, 128)
        save_image(out, f"generated/generated_{digit}_ex{example}.png")

    
for idx in range(10):
    inference(idx, num_examples=1)

if __name__=="__main__":
    
    pass
