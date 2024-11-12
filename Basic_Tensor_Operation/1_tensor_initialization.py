import torch

# setting up the device
device = torch.device("mps")
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)


print(my_tensor)
print("Shape: ", my_tensor.shape)
print("Dtype of Tensor: ", my_tensor.dtype)


# Empty Tensor Initialization
x = torch.empty((3, 3), dtype=torch.float32)
print(x)
