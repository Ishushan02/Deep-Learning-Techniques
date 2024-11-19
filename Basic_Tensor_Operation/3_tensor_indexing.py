import torch

batch = 10
features = 25

x = torch.rand((batch, features))

print(f"x is {x} and shape is {x.shape}")

print(f"x[0] is {x[0]}") 
print(f"x[0] is also x[0, :] {x[0, :]} and it's shape is {x[0, :].shape}")
print(f"x[:, 0] {x[:, 0]} and it's shape is {x[:, 0].shape}") # 1st feature or 1st column


# get 3rd example in the batch get it's 10 features
print(f"10 features of 3rd example is {x[2, :10]}")


# FANCY INDEXING

x = torch.arange(10)
indices = [2, 5, 8]
print(f"elemts at 2, 5 and 8 are {x[indices]}")

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])

print(f" x of rows of cols is {x[rows, cols]}")

# More advanced Indexing
x = torch.arange(10)
print(f"x is {x}")
print(f"All elements in x which is less than 4 or greater than 8 are {x[(x< 4) | (x > 8)]}")
print(f"All elements in x which is less than 4 and greater than 8 are {x[(x< 4) & (x > 8)]}")
print(f"All elements who is divisble by 2  {x[(x.remainder(2) == 0)]}")


# Some other useful Operations
print(f"Elements is x > 5 then OP x or else x * 2 is {torch.where(x > 5, x, x * 2)}")

print(f" Only Unique Elements {torch.tensor([0, 0, 1, 2, 3, 1,4, 4, 2, 5]).unique()}")
y = torch.rand((3, 5, 7))
print(f"How many dimension is y of {y.ndimension()}")

print(f"Number of Elements in X is {x.numel()}")