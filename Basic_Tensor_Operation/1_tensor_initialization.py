import torch

# setting up the device
device = torch.device("mps")
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)


print(my_tensor)
print("Shape: ", my_tensor.shape)
print("Dtype of Tensor: ", my_tensor.dtype)


# Empty Tensor Initialization
x = torch.empty((3, 3), dtype=torch.float32) # the values can be random and not zeros everytime
print(x)


x = torch.zeros((3, 3), dtype=torch.int)
print(x)

x = torch.rand((3, 3), dtype=torch.float)
print(x)

x = torch.ones((3, 3), dtype=torch.float)
print(x)

x = torch.eye(5, 5, dtype=torch.int32) # It gives out Identity Matrix
print(x)

x = torch.arange(start=0, end=100, step=2) # Generates values within a specified range with a fixed step size between values. x will be exclusive of 100(last value)
print(x) #(0, 2, 4, 6, .... 98)

x= torch.linspace(start= 0.1, end=1, steps=10) # Generates values between a start and an end point, with a fixed number of equally spaced points in the interval.
print(x)
# Use torch.linspace when you want a specific number of 
# values evenly distributed between a start and end point
# (often useful for smooth intervals or graphs).

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print("Empty Matrix with 0 Mean and 1 STD: ", x)

x = torch.empty(size=(1, 5)).uniform_(0, 1) # Creates Uniform Distribution with mean 0  and std 1
print("Empty Matrix with 0 Mean and 1 STD: ", x)


x = torch.diag(torch.ones(5))
print("Diagonal Matrix: ", x)

y = torch.arange(4)
print(y)
print(y.bool())
print(y.short()) # convert the dtype to int16