import torch
from sklearn import datasets
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


diabetes = datasets.load_diabetes()

# print(diabetes.data.shape, diabetes.target.shape)
X = diabetes.data
y = diabetes.target


transformer = StandardScaler()
X = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X Train Size: {X_train.shape} X Test Size:  {X_test.shape} Y Train Size: {y_train.shape} Y Test Size: {y_test.shape}")
print()

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
Y_train_tensor = torch.tensor(y_train)
Y_test_tensor = torch.tensor(y_test)

class LinearRegression(nn.Module):

    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1)

    
    def forward(self, x):
        return self.linear(x)
    
totalFeatures = X_train_tensor.shape[1]
# print(totalFeatures)
model = LinearRegression(input_size=totalFeatures)

lossFn = nn.MSELoss()
# lossFn = nn.HuberLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 10

print(X_train_tensor[0])
print(X_train_tensor[0].view(1, -1))

total_loss = []
for each_epoch in range(epochs):

    model.train()

    for i in range(X_train_tensor.shape[0]):
        epoch_losses = 0
        # each data points
        xtrain = X_train_tensor[i].view(1, -1)
        y_pred = model(xtrain)
        each_loss = lossFn(y_pred, Y_train_tensor[i].view(1, -1))
        each_loss.backward()
        optimizer.step()

        epoch_losses += each_loss.item()

        total_loss.append(epoch_losses /len(X_train_tensor) )

        
        print(f' Epoch [{each_epoch+1}/{epochs}], Loss: {total_loss[-1]:.4f}')




    p