import torch

x = torch.arange(9)

print(f"x is {x} and shape of x is {x.shape}")

x_3x3 = x.view(3, 3) # view stores the data contiguously in the memory
print(f"x_3x3 is rehsaped to {x_3x3} and it's shape is {x_3x3.shape}")

x_3x3 = x.reshape(3, 3)
print(f"x_3x3 is rehsaped to {x_3x3} and it's shape is {x_3x3.shape}")

y = x_3x3.T
print(f"Transpose of x_3x3 is {y}")


# converting 3 * 3 matrix to a linear vector
z = y.contiguous().view(9) # number of elements should be same as that we are reshaping the linear array

print(f" Converted 3 * 3 into linear z is {z}")


x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(f"x1 is {x1}")
print(f"x2 is {x2}")
print(f" Concatenating x1 and x2 along dim 0, {torch.concat((x1, x2), dim= 0)}")
print(f" Concatenating x1 and x2 along dim 1, {torch.concat((x1, x2), dim= 1)}")
print(f" Concatenating x1 and x2 , {torch.concat((x1, x2))}")


z = x1.view(-1) # Flattening the entire thing
print(f"{z} and it's size is {z.shape}")

batch = 60
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)

print(f"{z} and it's size is {z.shape}") # flattened

# switching the axes
z = x.permute(0, 2, 1) 

print(f"shape of x is {x.shape } and Z it's size is {z.shape}")

x = torch.arange(10)
# convert it to (1, 10)
y = x.unsqueeze(0) # 1, 10
z = x.unsqueeze(1) # 10, 1
k = x.unsqueeze(0).unsqueeze(1) # 1, 1, 10
p = x.unsqueeze(1).unsqueeze(0) # 1, 10, 1



print(f"x is {x} and it's shape is {x.shape}")
print(f"y is {y} and it's shape is {y.shape}")
print(f"z is {z} and it's shape is {z.shape}")
print(f"k is {k} and it's shape is {k.shape}")
print(f"p is {p} and it's shape is {p.shape}")





