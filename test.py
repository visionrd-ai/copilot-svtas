import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Change port if needed

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """A small dummy model."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    """Training function for each process."""
    setup(rank, world_size)

    model = SimpleModel().to(rank)  # Move model to assigned GPU
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy data
    inputs = torch.randn(4, 10).to(rank)
    targets = torch.randn(4, 10).to(rank)

    for epoch in range(5):  # Small loop for demonstration
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Get available GPUs
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs!")

    import torch
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(local_rank)
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
