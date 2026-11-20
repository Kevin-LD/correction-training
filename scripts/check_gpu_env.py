"""Run on the GPU container; no model downloads or training checkpoints needed."""
import argparse
from datetime import timedelta
import os

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--distributed', action='store_true')
args = parser.parse_args()
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 8, 'Expected eight visible GPUs'
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)

if args.distributed:
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([float(rank + 1)], device=f'cuda:{rank}')
        dist.all_reduce(value)
        assert value.item() == 36, value
        print(f'Rank {rank}: NCCL all_reduce passed', flush=True)
    finally:
        dist.destroy_process_group()
else:
    for i in range(torch.cuda.device_count()):
        device = f'cuda:{i}'
        props = torch.cuda.get_device_properties(i)
        print(i, props.name, f'{props.total_memory / 2**30:.1f} GiB', flush=True)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(256, 256, device=device, dtype=dtype, requires_grad=True)
            (x @ x.T).float().square().mean().backward()
            assert x.grad.isfinite().all().item()
        torch.cuda.synchronize(i)
    print('All eight GPUs passed FP32/BF16 forward and backward checks')
