import torch

size = 1000000
one = torch.ones(size, device="cuda")

x = torch.zeros(size, device="cuda")
y = torch.zeros(size, device="cuda")
x.add_(one)
y.add_(x)
print("x", x)
print("y", y)

s = torch.cuda.Stream()
x = torch.zeros(size, device="cuda")
y = torch.zeros(size, device="cuda")
for _ in range(10000):
    x.add_(one)
with torch.cuda.stream(s):
    y.add_(x)
print("x", x)
print("y", y)
