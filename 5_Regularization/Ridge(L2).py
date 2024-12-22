import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy

torch.manual_seed(42)


X = torch.randn(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)


class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input feature, 1 output
        
    def forward(self, x):
        return self.linear(x)
    

def ridgeLoss(model, output, target, ridge_coeff):
    mse_loss = nn.MSELoss()(output, target)
    penalty = torch.sum(torch.square(model.linear.weight))
    return mse_loss + ridge_coeff * penalty


modelA = LinearRegression()
modelB = LinearRegression()

ridge_coeff = 0.1

def train_ridge(model, X, y, lambda_ridge, epochs=100, learning_rate=0.01):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        
        # Calculate Ridge loss
        loss = ridgeLoss(model, predictions, y, lambda_ridge)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store the loss history
        loss_history.append(loss.item())
        
    return loss_history


def train_normal(model, X, y, epochs=100, learning_rate=0.01):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        
        # Calculate Ridge loss
        loss = nn.MSELoss()(predictions, y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store the loss history
        loss_history.append(loss.item())
        
    return loss_history


loss_history = train_ridge(modelA, X, y, lambda_ridge = ridge_coeff)
loss_history = train_normal(modelB, X, y)

with torch.no_grad():
    ridge_predictions = modelA(X)
    normal_predictions = modelB(X)

plt.figure(figsize=(10,6))
print(X.shape, y.shape, ridge_predictions.numpy().shape)
plt.scatter(X.numpy(), y.numpy(), label='Data', color='blue')
plt.plot(X.numpy(), ridge_predictions.numpy(), label='Lasso Fit', color='red', linewidth=2)
plt.plot(X.numpy(), normal_predictions.numpy(), label='Normal Fit', color='green', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Ridge Regression Fitted Lines')
plt.show()
