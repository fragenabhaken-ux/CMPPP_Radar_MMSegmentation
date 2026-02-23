import torch

# def get_vram(device):
#     alloc = torch.cuda.max_memory_allocated(device) / 1024**2
#     reserv = torch.cuda.max_memory_reserved(device) / 1024**2
#     total = torch.cuda.get_device_properties(device).total_memory / 1024**2
#     torch.cuda.reset_peak_memory_stats(device)
#     return alloc, reserv, total


# print("CUDA available:", torch.cuda.is_available())
# print("Torch CUDA version:", torch.version.cuda)
# print("CUDA device count:", torch.cuda.device_count())
# print()

# for i in range(torch.cuda.device_count()):
#     dev = torch.device(f"cuda:{i}")

#     x = torch.randn(4096, 4096, device=dev)
#     y = x @ x

#     torch.cuda.synchronize(dev)

#     alloc, reserv, total = get_vram(dev)

#     print(
#         f"cuda:{i} | "
#         f"allocated={alloc:.1f} MB | "
#         f"reserved={reserv:.1f} MB | "
#         f"total={total:.0f} MB"
#     )

def get_vram(device=None) -> tuple[float, float, float]:
    """
    Returns a tuple of (max_allocated_MB, max_reserved_MB, available_MB) GPU memory usage
    and resets the peak memory stats.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    if device is None:
        device = torch.device("cuda")

    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # Convert to MB
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)  # Convert to MB
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # Total VRAM in MB

    print("\n===== GPU VRAM Stats =====")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Max Allocated : {max_allocated:.2f} MB")
    print(f"Max Reserved  : {max_reserved:.2f} MB")
    print(f"Total Memory  : {total_memory:.2f} MB")
    print(f"Free (approx) : {total_memory - max_reserved:.2f} MB")
    print("==========================\n")

    # Reset the peak memory stats after reading
    torch.cuda.reset_peak_memory_stats(device)
    return max_allocated, max_reserved, total_memory