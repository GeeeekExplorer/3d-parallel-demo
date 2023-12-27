import os
import torch
import torch.distributed as dist
from torch.autograd import Function
from model import Net


class LinearWithAsyncComm(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input @ weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.t()
        handle = dist.all_reduce(grad_input, async_op=True)
        # input和output可能是多维，但weight肯定是二维
        grad_weight = input.t().view(weight.size(0), -1) @ grad_output.view(-1, weight.size(1))
        handle.wait()
        return grad_input, grad_weight


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        dist.all_reduce(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TPNet(Net):
    def forward(self, x):
        x = LinearWithAsyncComm.apply(x, self.w1)
        x = x.relu() @ self.w2
        AllReduce.apply(x)
        return x

    @classmethod
    def to_tp(cls, module: Net, world_size, rank):
        device = next(module.parameters()).device
        in_dim, out_dim, hidden_dim = module.w1.size(0), module.w2.size(1), module.w1.size(1)
        tp_module = cls(in_dim, out_dim, hidden_dim // world_size).to(device)
        tp_module.w1.data.copy_(module.w1.data.chunk(world_size, dim=1)[rank])
        tp_module.w2.data.copy_(module.w2.data.chunk(world_size, dim=0)[rank])
        return tp_module


if __name__ == '__main__':
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    net = Net(64, 10, 128).cuda()
    X = torch.randn(32, 64, device="cuda")
    Y = net(X)
    Y.mean().backward()
    print(Y[:, -1])
    # print(net.w1.grad)

    net = TPNet.to_tp(net, dist.get_world_size(), dist.get_rank())
    Y = net(X)
    Y.mean().backward()
    print(Y[:, -1])
    # print(net.w1.grad)
