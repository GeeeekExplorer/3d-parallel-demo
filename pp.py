import os
import torch
import torch.distributed as dist
from model import Net


class Send(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, rank):
        ctx.rank = rank
        dist.send(output, rank + 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        dist.recv(grad_output, rank + 1)
        return grad_output, None


class Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rank):
        ctx.rank = rank
        dist.recv(input, rank - 1)
        return input

    @staticmethod
    def backward(ctx, grad_input):
        rank = ctx.rank
        dist.send(grad_input, rank - 1)
        return grad_input, None


class Pipe(torch.nn.Module):
    def __init__(self, module: torch.nn.Sequential, shape, chunks=1):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.is_first = self.rank == 0
        self.is_last = self.rank == self.world_size - 1
        size = len(module) // self.world_size
        offset = size * self.rank
        self.module = module[offset:offset+size]
        self.chunks = chunks
        shape = list(shape)
        shape[0] //= chunks
        self.shape = shape

    def forward(self, x: torch.Tensor):
        ys = []
        xs = x.chunk(self.chunks)
        for x in xs:
            if not self.is_first:
                x = x.new_empty(self.shape).requires_grad_()
                x = Recv.apply(x, self.rank)
            y = self.module(x)
            if not self.is_last:
                y = Send.apply(y, self.rank)
            ys.append(y)
        return torch.cat(ys)


if __name__ == '__main__':
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    num_layers, in_dim, out_dim, hid_dim, inter_dim = 8, 64, 10, 128, 256
    bs, chunks = 32, 8
    layers = []
    layers.append(Net(in_dim, hid_dim, inter_dim))
    for _ in range(num_layers - 2):
        layers.append(Net(hid_dim, hid_dim, inter_dim))
    layers.append(Net(hid_dim, out_dim, inter_dim))

    net = torch.nn.Sequential(*layers).cuda()
    X = torch.randn(bs, in_dim, device="cuda")
    Y = net(X)
    Y.mean().backward()
    print(Y[:, -1])
    print(net[0].w1.grad)
    net.zero_grad()

    net = Pipe(net, (bs, hid_dim), chunks).cuda()
    Y = net(X)
    if net.is_last:
        Y.mean().backward()
    else:
        torch.autograd.backward(Y, torch.empty_like(Y))
    if net.is_last:
        print(Y[:, -1])
    if net.is_first:
        print(net.module[0].w1.grad)
