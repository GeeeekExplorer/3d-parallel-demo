import os
import torch
import torch.distributed as dist
from torch import Tensor
from model import Net


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
        self.register_buffer("buf", torch.empty(*shape))

    def forward(self, x: Tensor):
        ys = []
        if self.is_first:
            xs = x.chunk(self.chunks)
        for i in range(self.chunks):
            if self.is_first:
                x = xs[i]
            else:
                dist.recv(self.buf, self.rank - 1)
                x = self.buf
            y = self.module(x)
            if self.is_last:
                ys.append(y)
            else:
                dist.send(y, self.rank + 1)
        if self.is_last:
            return torch.cat(ys)


if __name__ == '__main__':
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    num_blocks, in_dim, out_dim, hid_dim, inter_dim = 8, 64, 10, 128, 256
    blocks = []
    blocks.append(Net(in_dim, hid_dim, inter_dim))
    for _ in range(num_blocks - 2):
        blocks.append(Net(hid_dim, hid_dim, inter_dim))
    blocks.append(Net(hid_dim, out_dim, inter_dim))
    net = torch.nn.Sequential(*blocks).cuda()
    X = torch.randn(32, 64, device="cuda")
    Y = net(X)
    print(Y.size(), Y[:, -1])

    net = Pipe(net, (32, 128), 1).cuda()
    Y = net(X)
    if net.is_last:
        print(Y.size(), Y[:, -1])