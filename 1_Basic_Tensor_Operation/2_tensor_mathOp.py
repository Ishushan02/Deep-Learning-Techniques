import torch
import numpy


x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)

# similarily for subtraction
z = x - y
print(z)


z = torch.true_divide(x, y)
# ELement Wise division so x and y must of same length
print(" This is Element by division of x and y", z)


# VERY IMPORTANT INPLACE OPERATIONS
t = torch.ones(3)
t.add_(x) # the tensor of t is added to x, hence it is more efficient and inplace operation takes place
print("InPlace operation any operation followed by underscore _ ", t) # t += x similar operation


# Exponentiation
z  =x.pow(2) # z = x ** 2
print("Element by power of 2: ", z)

# similarily z > 0 or z , 0

# Matrix Multiplication

x1 = torch.rand(2, 5)
x2 = torch.rand(5, 2)

z = torch.mm(x1, x2)
print(f"Matrix multiplication of {x1} and {x2} is {z}")


z = torch.matmul(x1, x2)
print(f"Matrix multiplication of {x1} and {x2} is {z}")


z = x1.mm(x2)
print(f"Matrix multiplication of {x1} and {x2} is {z}")


matrix_exp = torch.rand(5, 5)
print(" Initial Metrix: ", matrix_exp)
matrix_exp.matrix_power(3)
print(" After Metrix Exponent of 3 : ", matrix_exp)



# DOT PRODUCT
x = torch.ones(3)
y = torch.rand(3)

print(" Dot Product: ", torch.dot(x, y))


# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
tensor3 = torch.rand((batch, p, m))

out_batch_mm = torch.bmm(tensor1, tensor2)

print(tensor2.shape)

tensor4 = tensor3.permute(0, 2, 1) # to reshape into desired shape

print(tensor4.shape)

# print(torch.matmul(tensor1, tensor4))
'''
Conclusion for batch Matrix Multiplication A = (batch, m, n) 
                                       and B = (batch, x, y)
                                         n and x  must be same.. 
 
'''


# VVVIIIII Conept of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand(1, 5)

print(f"x1 {x1} and x2 {x2}")

z = x1 - x2 # what x2 will do is it will be broadcasted to x1
print(f"x1 - x2 bradcasted in shape of x1  is {z}")

z = x1 ** x2
print(f"x1 element wise power to x2 is  {z}")

x1 = torch.rand(size=(5, 2, 3), dtype=torch.float64)
print(f"x1 is {x1} ")

print(f"Sum of along 0th dim:  {torch.sum(x1, dim=0)}")
print(f"Sum of along 1st dim:  {torch.sum(x1, dim=1)}")
print(f"Sum of along 2nd dim:  {torch.sum(x1, dim=2)}")

'''
Example 

x1 = [

    [[1, 2, 3],
    [4, 5, 6]],
    
    [[7, 8, 9],
    [10, 11, 12]],

    [[13, 14, 15],
    [16, 17, 18]],

    [[19, 20, 21],
    [22, 23, 24]],

    [[25, 26, 27],
    [28, 29, 30]],

]


sum along dim 0

    sum of all (0, 0): 1 + 7 + 13 + 19 + 25 = 65

    sum of all (0, 1): 2 + 8 + 14 + 20 + 26 = 70

    sum of all (1, 2): 3 + 9 + 15 + 21 + 27 = 90

    similarily the second rows 

    hence sum across dim 0 is  = [[65, 70, 90],
                                  [80, 85, 90]]


sum along dim 1 

    [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
    [7, 8, 9] + [10, 11, 12] = [17, 19, 21]
    similarily all other batches
    .....

    hence sum across dim 1 is = [
                                [5, 7, 9],
                                [17, 19, 21],
                                [29, 31, 33],
                                [41, 43, 45],
                                [53, 55, 57]
                                ]


sum along dim 2

    In 1st matrix
    Row 1 = 1 + 2 + 3
    Row 2 = 4 + 5 + 6

    In 2nd matrix
    Row 1 = 7 + 8 + 9
    Row 2 = 10 + 11 + 12

    .. similarily so on 

    hence sum across dim 2 is =[
                               [6, 15],
                               [24, 33],
                               [42, 51],
                               [60, 69],
                               [78, 87]
                               ]


'''

x = torch.tensor([12, 100, 4, 12, 34, 54, 89])

val, index = torch.max(x, dim=0)
print(f"max Val and it's max index is {val}, {index}")
val, index = torch.min(x, dim=0)
print(f"min Val and it's min index is {val}, {index}")

abs_x = torch.abs(x)
print(f"abs of x is {abs_x}")

arg_max = torch.argmax(x, dim=0)
print(f"ArgMax of x is {arg_max}") # index of the maximum number in the tensor

arg_min = torch.argmin(x, dim=0)
print(f"ArgMin of x is {arg_min}") # index of the maximum number in the tensor

meanX = torch.mean(x.float())
print(f"Mean of x is {meanX}")

y = torch.tensor([32, 76, 11, 12, 56, 54, 89])
print(f" y is {y}")
z = torch.eq(x, y)

print(f" z {z} Which elemts are equal in x and y")

sorted_y, indices = torch.sort(y, dim = 0)
print(f" Sort torch sorted y is {sorted_y}")


tensor = torch.tensor([-1, 4, 6, -9, - 55, 33, 7, -11, -99, 100])

z = torch.clamp(tensor, min = -3, max = 30)

# all value less than or equal to -3 will be equalized to -3
# all value more than or equal to 30 will be equalized to 30
# rest will be as it is

print(f" Clamp makes all the rest element which are not in it's min and max range {z}")


x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x) # The integer values 1 are interpreted as True and 0 as False. So even 1 is true it will return tru
print(f" Any function {z}")
z = torch.all(x)
print(f" All function {z}") # return true if all elemets are true

