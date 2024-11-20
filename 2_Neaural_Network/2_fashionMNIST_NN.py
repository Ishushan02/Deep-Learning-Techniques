import torch
from torch import nn
from torch.nn import functional as Fn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets 
import cv2



class Architecture(nn.Module):

    def __init__(self, input_shape, classes):
        super(Architecture, self).__init__()
        self.fc1 = nn.Linear(input_shape, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, classes)
        

    def forward(self, x):
        out1 = Fn.relu(self.fc1(x))
        out2 = Fn.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out
    
batch_size = 128
learning_rate = 0.001
input_shape = 784
num_classes = 10
epochs = 100


device = torch.device("cpu")

model = Architecture(input_shape=input_shape, classes=num_classes)
model.to(device)

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = datasets.FashionMNIST("/Users/ishananand/Desktop/Pytorch/", train = True, transform=transforms.ToTensor(), download=True)
train_data_Loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print(train_data.data.shape)

test_data = datasets.FashionMNIST("/Users/ishananand/Desktop/Pytorch/", train = False, transform=transforms.ToTensor(), download=True)
test_data_Loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(test_data.data.shape)

'''
# Training the Data
for each_epoch in range(epochs):
    num_samples = 0
    correct_pred = 0
    epoch_loss = 0
    for batch_id, (trainX, trainY) in enumerate(train_data_Loader):
        trainX = trainX.to(device)
        trainX = trainX.view(trainX.shape[0], -1)
        trainY = trainY.to(device)

        output = model(trainX)
        # print(output.shape)
        loss_data = loss(output, trainY)

        optimizer.zero_grad()
        loss_data.backward()
        optimizer.step()

        epoch_loss += loss_data.item()  # Add batch loss to epoch loss
        with torch.no_grad():  # No gradient computation for accuracy
            predictions = torch.argmax(output, dim=1)  # Get predicted class labels
            correct_pred += (predictions == trainY).sum().item()  # Count correct predictions
            num_samples += trainY.size(0)  # Update total number of samples
        # break
        # print(len(train_data_Loader), num_samples)
    print(f"{each_epoch + 1}/{epochs} and loss is {epoch_loss/len(train_data_Loader)} and accuracy is {correct_pred/num_samples * 100}")


torch.save(model.state_dict(), f = "/Users/ishananand/Desktop/Pytorch/Model/fashionmnistModel.pth")
print(f"Model saved to Model Path")

'''

test_model = Architecture(input_shape=input_shape, classes=num_classes)
# Load the saved model weights
test_model.load_state_dict(torch.load('/Users/ishananand/Desktop/Pytorch/Model/fashionmnistModel.pth'))

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


testAccuracy = testAccuracy(test_model, test_data_Loader)
print(f"Test Accuracy is  {testAccuracy}")

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

