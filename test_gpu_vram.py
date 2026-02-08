import torch

def get_vram(device):
    alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    reserv = torch.cuda.max_memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    return alloc, reserv, total


print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print()

for i in range(torch.cuda.device_count()):
    dev = torch.device(f"cuda:{i}")

    x = torch.randn(4096, 4096, device=dev)
    y = x @ x

    torch.cuda.synchronize(dev)

    alloc, reserv, total = get_vram(dev)

    print(
        f"cuda:{i} | "
        f"allocated={alloc:.1f} MB | "
        f"reserved={reserv:.1f} MB | "
        f"total={total:.0f} MB"
    )