import os
import torch
import torch.distributed as dist
from torch import Tensor
from model import Net


class DDP(torch.nn.Module):
    MIN_BUCKET_SIZE = 1024 * 1024

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.buckets: list[Tensor] = []                        # bucket用于保存梯度和同步
        self.comm_stream = torch.cuda.Stream()

        num_params = len(list(module.parameters()))
        bucket_params: list[Tensor] = []                       # 一个bucket对应的参数列表
        bucket_size = 0

        for idx, param in enumerate(reversed(list(module.parameters()))):
            if not param.requires_grad:
                continue
            bucket_size += param.numel()
            bucket_params.append(param)
            if bucket_size < DDP.MIN_BUCKET_SIZE and idx + 1 < num_params:
                continue
            # 攒满bucket或者已经是最后一个参数
            bucket = bucket_params[0].new_zeros(bucket_size)
            bucket.ready = False
            offset = 0
            for param in bucket_params:
                param.grad = bucket[offset:offset+param.numel()].view_as(param)
                offset += param.numel()
                param.register_post_accumulate_grad_hook(self.make_hook(param, bucket, bucket_params))
                param.ready = False
            self.buckets.append(bucket)
            bucket_params = []
            bucket_size = 0
    
    def make_hook(self, param: Tensor, bucket: Tensor, bucket_params):
        def hook(*args):
            param.ready = True
            if all(p.ready for p in bucket_params):
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    dist.all_reduce(bucket, dist.ReduceOp.AVG)
                bucket.ready = True

            if all(b.ready for b in self.buckets):
                torch.cuda.current_stream().wait_stream(self.comm_stream)
                for p in self.module.parameters():
                    if p.requires_grad:
                        p.ready = False
                for b in self.buckets:
                    b.ready = False
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    net = Net(64, 10, 128).cuda()
    X = torch.randn(32, 64, device="cuda", requires_grad=True)
    Y = net(X)
    Y.mean().backward()
    print(Y[:, -1])
    # print(net.w1.grad)

    net = DDP(net)
    X = X.chunk(dist.get_world_size())[dist.get_rank()]
    Y = net(X)
    Y.mean().backward()
    print(Y[:, -1])
    # print(net.module.w1.grad)
