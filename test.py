import torch

a = torch.tensor([[1,2,3],[3,4,1]])
b = torch.tensor([[1,3,3],[1,4,2]])

c = a.eq(b)
print(c.sum().item())