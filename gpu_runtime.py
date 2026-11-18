"""Make the project environment's CUDA libraries available on native Windows."""
import os
import sys
from pathlib import Path


def configure_gpu():
    cuda_bin = Path(sys.prefix) / "Library" / "bin"
    os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ["PATH"]
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "TensorFlow cannot see the NVIDIA GPU. Run run-training.cmd to use "
            "this project's .venv and CUDA libraries."
        )
    device = tf.config.experimental.get_device_details(gpus[0])
    print(f"TensorFlow {tf.__version__} | GPU: {device['device_name']}", flush=True)
