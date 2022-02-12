import torch
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# my_tensor = torch.tensor([[1, 2, 3],
#                           [4, 5, 6]], dtype = torch.float32,
#                           device = device,
#                           requires_grad = True)
#
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)


# x = torch.zeros((3, 3))
# print(x)
#
# y = torch.rand((3, 3))
# print(y)
#
# z = torch.ones((5, 5))
# print(z)
#
# I = torch.eye(3,3, dtype = torch.int32)
# print(I)
#
# b = torch.arange(0, 10, 1)
# print("arange", b)
#
# lins = torch.linspace(start = 0.1, end = 1, steps = 10)
# print(lins)

# gauss = torch.empty((3, 4)).normal_(mean = 2, std = 5)
# print(gauss)
#
# unif = torch.empty((3, 4)).uniform_(0, 1)
# print(unif)
#
# diag = torch.diag(torch.tensor([5, 6, 8, 10]))
# print(diag)

#   How to initialize and convert tensor to other types
# tensor = torch.arange(5)
# print(tensor.bool())
# print(tensor.short())   #       int16
# print(tensor.long())    #       int64
# print(tensor.half())    #       float16
# print(tensor.float())   #       float32
# print(tensor.double())   #       float64

#   Array to Tensor conversion and vice-versa
# import numpy as np
# np_arr = np.zeros((5, 5))
# #   Convert numpy array to tensor
# tensor = torch.from_numpy(np_arr)
# print(tensor)
# #   Convert tensor to numpy arr
# np_arr_back = tensor.numpy()
# print(np_arr_back)

#  ===========================================================
#           Tensor Math & Comparison Operators
#  ==========================================================

# x = torch.tensor([1, 2, 3])
# y = torch.tensor([9, 8, 7])

#   Division
# z = torch.true_divide(x, 2)
# print(z)

#   Matrix multiplication
# a = torch.randint(0, 5, (2, 5))
# print(a)
# b = torch.randint(0, 5, (5, 3))
# print(b)
# print(a.mm(b))

#   Link: https://medium.com/swlh/interesting-ways-to-work-with-tensors-in-pytorch-fce6203f7388
#   Matrix exponentiation
# torch.manual_seed(2)
# mtr_exp = torch.randint(0, 5, (3, 3))
# print(mtr_exp)
# print(torch.matrix_power(mtr_exp, 3))


#   Matrix inverse
# a = torch.tensor([[.1, .2, .3], [.5, .4, .3], [.2, .5, .6]])
# mtr_inv = torch.inverse(a)
# print(mtr_inv)


#       EigenVector, EigenValue
# mtr = torch.tensor([[5, 2], [9, 2]], dtype = torch.float32)
# eig = torch.eig(mtr, eigenvectors = True)
# print(eig)


#   Matrix rank
# a = torch.tensor([[5, 2], [9, 2]], dtype = torch.float32)
# print(torch.matrix_rank(a))


# x = torch.randint(0, 5, (1, 2))
# print(x)
# y = torch.randint(0, 5, (2, 1))
# print(y)
# z = torch.dot(x, y)


#   Batch Matrix Multiplication
# batch = 32
# m = 5
# n = 3
# p = 4
# tensor_01 = torch.randn((batch, m, n))
# tensor_02 = torch.randn((batch, n, p))
#
# out_batch = torch.bmm(tensor_01, tensor_02)
# print(out_batch.shape)

#   Broadcasting
# x1 = torch.randint(0, 5, (3, 3))
# print(x1)
# x2 = torch.randint(0, 5, (1, 3))
# print(x2)
# print(x1 ** x2)


#   Other useful tensor operators
x1 = torch.randint(-255, 255, (3, 3))
x2 = torch.randint(-5, 5, (3, 3))

# sum = torch.sum(x1, dim = 1)
# val, inds = torch.max(x1, dim = 0)
# val, inds = torch.min(x1, dim = 1)
# abs_x = torch.abs(x1)
#
# z = torch.argmax(x1, dim = 0)
# z = torch.argmin(x1, dim = 0)
#
# mean = torch.mean(x1.float(), dim = 0)
#
# #   Check 2 matrix whether they are equl
# mtr_equal = torch.eq(x1, x2)
#
# #   Sort mtr
# val, index = torch.sort(x1, dim = 0, descending = False)

z = torch.clamp(x1, min = 0, max = 255)

z = torch.tensor([1, 0, 2, 0, 1, 1])
# print(torch.all(z))


#   ===============================================================
#                           Tensor Indexing
#   ================================================================
# y = torch.randint(0, 10, (5, 5))
# indices = [2, 5, 8]
# print(y)
# print(y[indices])

# z = torch.arange(10)
# print(z[(y<2) & (z>7)])

# rows = torch.tensor([0, 1])
# cols = torch.tensor([2, 4])
# print(y[rows, cols])

# a = torch.randint(0, 10, (3, 2))
# print(a)
# x, y = torch.where(a>5)
# print(x)
# print(y)
# print(a[x, y])

# print(x.ndimension())
# print(x.unique())
# print(x.numel())

#   ===============================================================
#                           Tensor Reshaping
#   ================================================================

# x = torch.randint(0,10,(2, 5))
# print(x.shape)
# z = x.reshape(-1)
# print(z.shape)

batch = 64
x = torch.rand(batch, 2, 5)
z = x.reshape(batch, -1)
print(z.shape)

a = torch.rand(2, 5)
z = x.permute(0, 2, 1)
print(z.shape)
z1 = z.unsqueeze(0)
print(z1.shape)




