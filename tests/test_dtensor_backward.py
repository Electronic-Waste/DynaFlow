import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
   ColwiseParallel,
   RowwiseParallel,
   parallelize_module,
)


class Model(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.up = torch.nn.Linear(10, 20)
      self.down = torch.nn.Linear(20, 10)

   def forward(self, x):
      return self.down(self.up(x))

def custom_backend(gm: torch.fx.GraphModule, *args, **kwargs):
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(gm.graph.python_code(root_module="self").src)
    return gm

torch._dynamo.config.compiled_autograd = True
@torch.compile(backend=custom_backend)
def train(model, x):
   loss = model(x).sum()
   loss.backward()

def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = Model().to(f"cuda:{local_rank}")
    if world_size > 1 and torch.cuda.is_available():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
        model = parallelize_module(model, mesh, {"up": ColwiseParallel(), "down": RowwiseParallel()})
    model = torch.compile(model, backend=custom_backend)
    x = torch.randn(10, device=f"cuda:{local_rank}")
    train(model, x)

if __name__ == "__main__":
    main()
