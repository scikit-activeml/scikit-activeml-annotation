import logging
import sys


try:
    # Optional depdendency
    import torch  # pyright: ignore[reportMissingImports]
except ImportError as e:
    logging.error(e)
    sys.exit(1)

if __name__ == '__main__':
    logging.info("PyTorch Version:", torch.__version__)
    logging.info("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        logging.info("CUDA is available!")
        logging.info("GPU:", torch.cuda.get_device_name(0))
    else:
        logging.info("CUDA is not available.")
