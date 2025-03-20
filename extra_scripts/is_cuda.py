
import torch


if __name__ == '__main__':
    print("PyTorch Version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")



