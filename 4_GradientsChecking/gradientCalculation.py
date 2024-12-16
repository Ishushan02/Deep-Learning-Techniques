import torch

input = torch.tensor([5], dtype=torch.float32)
input.requires_grad = True


def f(x):
    return x ** 2; # f(x) = x^2

# d(f(x)/d(x)) = 2 * x

output = f(input)
print(output)

'''
Calling Backward on tensor will tell Pytorch to use the computational
graph to calculate partial derivative of this value w.r.t all parameters 
requiring a gradient that are into computaional graph evaluated at parameters
current Value
'''
print(output.backward())
print(input.grad) 



ex = torch.tensor([10], dtype=torch.float32)
ex.requires_grad = True
out = f(ex)
print(out)
out.backward()
print(ex.grad)