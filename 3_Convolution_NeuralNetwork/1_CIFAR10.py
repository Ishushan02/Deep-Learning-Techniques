import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as Fn
import torch.optim as optim


class CNNArchitecture(nn.Module):
    def __init__(self, num_inChannels, num_Classes):
        super(CNNArchitecture, self).__init__()

        self.con2d1 = nn.Conv2d(in_channels=num_inChannels, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.maxPool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.con2d2 = nn.Conv2d(in_channels=16, out_channels= 32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.fc1 = nn.Linear(8 * 4 * 4, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_Classes)

    def forward(self, x):

        self.conv1out = Fn.relu(self.con2d1(x))
        self.maxout1 = self.maxPool(self.conv1out)
        self.conv2out = Fn.relu(self.con2d2(self.maxout1))
        self.maxout2 = self.maxPool(self.conv2out)

        self.flatten = self.maxout2.reshape(self.maxout2.shape[0], -1)#Flattening 
        # print("+++++++++",self.flatten.shape)
        self.fcout = Fn.relu(self.fc1(self.flatten))
        out = self.fc2(self.fcout)

        return out
    

# model = CNNArchitecture(3, 10)
# img = torch.randn(64, 3, 28, 28)
# print(model(img).shape)
# CNNArchitecture(3, 10)

batchSize = 64

# CIFAR10 Mean. and Std.
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# Train and test transformations with normalization
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Normalize with the given mean and std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Normalize with the same values for consistency
])

trainData = datasets.CIFAR10("/Users/ishananand/Desktop/Pytorch/dataset", train=True, transform=train_transform, download=True)
trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)

testData = datasets.CIFAR10("/Users/ishananand/Desktop/Pytorch/dataset", train=False, transform=test_transform, download=True)
testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

device = torch.device("cpu")
epochs = 500
model = CNNArchitecture(num_inChannels=3, num_Classes=10)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
lossFn = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('/Users/ishananand/Desktop/Pytorch/Model/cifar10Model.pth'))


for each_epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0 
    total_samples = 0

    for batch_id, (trainX, trainY) in enumerate(trainDataLoader):

        trainX = trainX.to(device)
        trainY = trainY.to(device)

        
        #forward
        pred = model(trainX)
        # print(pred.shape, trainY.shape)
        predy = torch.argmax(pred, dim=1)
        # break

        # backward
        lossval = lossFn(pred, trainY)
        optimizer.zero_grad()

        # grad descent
        lossval.backward()

        optimizer.step()
        epoch_loss += lossval.item()  # Add batch loss to epoch loss
        with torch.no_grad():  # No gradient computation for accuracy
            # predictions = torch.argmax(pred, dim=1)  # Get predicted class labels
            correct_predictions += (predy == trainY).sum().item()  # Count correct predictions
            total_samples += trainY.size(0)  # Update total number of samples
    average_loss = epoch_loss / len(trainDataLoader)  # Average loss
    accuracy = correct_predictions / total_samples * 100  # Accuracy as percentage

    # Display metrics for the epoch
    print(f"Epoch {each_epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}%")


        

torch.save(model.state_dict(), f = "/Users/ishananand/Desktop/Pytorch/Model/cifar10Model.pth")
print(f"Model saved to Model Path")



# Load the saved model weights
# test_model.load_state_dict(torch.load('/Users/ishananand/Desktop/Pytorch/Model/mnistModel.pth'))

# Move the model to the appropriate device (CPU or GPU)
# 5. Set the model to evaluation mode (important for inference)

# the model to evaluation mode (important for inference)
model.eval()

def testAccuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = x.reshape(x.shape[0], -1)
            # print(x.shape)

            scores = model(x)

            predictions = torch.argmax(scores, dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples * 100  # Accuracy as percentage
    return accuracy


testAccuracy = testAccuracy(model, testDataLoader)
print(f"Test Accuracy is  {testAccuracy}")
