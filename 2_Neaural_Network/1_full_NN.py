import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2

'''
All the dataset documentation  of Pytorch : https://pytorch.org/vision/main/datasets.html
'''

# Creating a Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self, input_size, classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, out_features=70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, classes)
    
    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out
    
# Let's test it with Random initializer
# model = NN(768, 10)
# x = torch.randn(64, 768) # (batches, )
# print(model(x).shape) #out put should be of 10 classes


device = torch.device("cpu")

# Parameter Initialization
input_size = 784
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epochs = 10


# Loading the Dataset Mnist
train_data = datasets.MNIST(root = '/Users/ishananand/Desktop/Pytorch/2_Neaural_Network/', train=True, transform=transforms.ToTensor(), download=True)
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root = '/Users/ishananand/Desktop/Pytorch/2_Neaural_Network/', train=False, transform=transforms.ToTensor(), download=True)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f"Train Dataset is {train_data.data.shape}")
print(f"Test Dataset is {test_data.data.shape}")

# Initialize the Network
model = NN(input_size=input_size, classes=num_classes).to(device)

# Loss and Optimizers
lossfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train the Network, 1 epoch means the network has seen all the datasets in the images
for each_epoch in range(100):
    epoch_loss = 0.0  # Accumulate loss for the epoch
    correct_predictions = 0  # Count correct predictions
    total_samples = 0  # Count total samples it is generally batch size at each epoch to calculate average

    # for each batch of data writing enumerate so that we can access each batchID for each batches
    for batch_id, (data, target) in enumerate(train_dataLoader): 
        data = data.to(device)
        target = target.to(device)
        # print(f"Batch {batch_id}, Data {data.shape} and target {target.shape}")
        # (Data Shape is (batch, Channel, Height, Width))
        data = data.reshape(data.shape[0], -1) # Flattenming and keeping the batches same

        # forward iteration
        scores = model(data)
        loss = lossfn(scores, target)

        # backward
        optimizer.zero_grad() # Initializing 0 gradients initially, it clears the gradients of  all the params that optim is responsible for updating
        loss.backward()

        # gradient descent Step
        optimizer.step() # updating weights

        epoch_loss += loss.item()  # Add batch loss to epoch loss
        with torch.no_grad():  # No gradient computation for accuracy
            predictions = torch.argmax(scores, dim=1)  # Get predicted class labels
            correct_predictions += (predictions == target).sum().item()  # Count correct predictions
            total_samples += target.size(0)  # Update total number of samples

    # Compute average loss and accuracy for the epoch
    average_loss = epoch_loss / len(train_dataLoader)  # Average loss
    accuracy = correct_predictions / total_samples * 100  # Accuracy as percentage

    # Display metrics for the epoch
    print(f"Epoch {each_epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}%")



torch.save(model.state_dict(), f = "/Users/ishananand/Desktop/Pytorch/2_Neaural_Network/Model/mnistModel.pth")
print(f"Model saved to Model Path")



test_model = NN(input_size=input_size, classes=num_classes)
# Load the saved model weights
test_model.load_state_dict(torch.load('/Users/ishananand/Desktop/Pytorch/2_Neaural_Network/Model/mnistModel.pth'))

# Move the model to the appropriate device (CPU or GPU)
# 5. Set the model to evaluation mode (important for inference)
test_model.to(device)

# the model to evaluation mode (important for inference)
test_model.eval()

def testAccuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            # print(x.shape)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples * 100  # Accuracy as percentage
    return accuracy


# testAccuracy = testAccuracy(test_model, test_dataLoader)
# print(f"Test Accuracy is  {testAccuracy}")
def testnewImg(imgPath, test_model):
    testImage = cv2.imread(imgPath, 0)
    testimgreshaped = cv2.resize(testImage, (28, 28))

    img = torch.tensor(testimgreshaped, dtype=torch.float32)
    print(testImage.shape, "-- Reshaped Image ", testimgreshaped.shape)
    print(img.shape)

    img = img.unsqueeze(0)
    print(img.shape)

    flattened_testimage = img.view(img.shape[0], -1)
    test_score = test_model(flattened_testimage)
    val, idx = test_score.max(1)
    return  idx

testnewImg("/Users/ishananand/Desktop/Pytorch/2_Neaural_Network/test_image.png", test_model)