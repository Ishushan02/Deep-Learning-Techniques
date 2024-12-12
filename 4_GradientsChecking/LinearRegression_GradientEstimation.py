import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Normalize the features (important for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for regression

    def forward(self, x):
        return self.linear(x)

# Function for training with different gradient descent strategies
def train_model(model, optimizer, criterion, X_train, y_train, batch_size=None, num_epochs=100):
    # For Mini-batch and Stochastic, we will shuffle the data
    if batch_size:
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        data_loader = [(X_train, y_train)]  # For batch gradient descent

    loss_list = []
    accuracy_list = []  # To store R² scores

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Collect predictions and targets for R² score calculation
            all_preds.append(y_pred.detach().numpy())
            all_targets.append(batch_y.detach().numpy())

        # Flatten the lists
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate R² score (accuracy for regression)
        r2 = r2_score(all_targets, all_preds)
        loss_list.append(epoch_loss / len(data_loader))
        accuracy_list.append(r2)
        print(f"Epoch [{epoch+1}/{num_epochs}], Iteration Loss: {loss.item():.4f}, R² Score: {r2:.4f}")

    return loss_list, accuracy_list

# Train the model using different gradient descent methods

# Initialize model
input_dim = X_tensor.shape[1]
model_sgd = LinearRegressionModel(input_dim)
model_bgd = LinearRegressionModel(input_dim)
model_mbgd = LinearRegressionModel(input_dim)

# Criterion (Mean Squared Error Loss)
criterion = nn.MSELoss()
# criterion = nn.HuberLoss()

# Stochastic Gradient Descent (SGD)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
loss_sgd, accuracy_sgd = train_model(model_sgd, optimizer_sgd, criterion, X_tensor, y_tensor, batch_size=1, num_epochs=100)

# Batch Gradient Descent (BGD)
optimizer_bgd = optim.SGD(model_bgd.parameters(), lr=0.01)
loss_bgd, accuracy_bgd = train_model(model_bgd, optimizer_bgd, criterion, X_tensor, y_tensor, batch_size=None, num_epochs=100)

# Mini-Batch Gradient Descent (MBGD)
optimizer_mbgd = optim.SGD(model_mbgd.parameters(), lr=0.01)
loss_mbgd, accuracy_mbgd = train_model(model_mbgd, optimizer_mbgd, criterion, X_tensor, y_tensor, batch_size=32, num_epochs=100)

# Plotting the loss curves
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(loss_sgd, label="SGD Loss")
plt.plot(loss_bgd, label="BGD Loss")
plt.plot(loss_mbgd, label="MBGD Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curves for different Gradient Descent Methods')

# Accuracy (R² score) plot
plt.subplot(1, 2, 2)
plt.plot(accuracy_sgd, label="SGD R²", color='r')
plt.plot(accuracy_bgd, label="BGD R²", color='g')
plt.plot(accuracy_mbgd, label="MBGD R²", color='b')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.legend()
plt.title('R² Score curves for different Gradient Descent Methods')

plt.tight_layout()
plt.show()
