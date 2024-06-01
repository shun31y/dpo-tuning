import torch


def measure_gpu_memory_allocation() -> float:
    bytes = torch.cuda.max_memory_allocated(device="cuda")
    return bytes / 1024**2  # MB


def measure_gpu_memory_for_load_model(num_parameters: int, load_bit: int) -> float:
    m = (num_parameters * 4) * 1.2 / (32 / load_bit)
    return m / 1024**2  # MB


def gpu_information() -> None:
    print("-----------------")
    print("device_name = {}".format(torch.cuda.get_device_name()))
    print("-----------------")
    print("is_available = {}".format(torch.cuda.is_available()))
    print("-----------------")
    print("device_count = {}".format(torch.cuda.device_count()))
    print("-----------------")
