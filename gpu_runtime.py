"""Make CUDA libraries available before TensorFlow loads on native Windows."""
import os
import sys
from pathlib import Path


def cuda_bin_dir():
    environment_bin = Path(sys.prefix) / "Library" / "bin"
    if (environment_bin / "cudart64_110.dll").is_file():
        return environment_bin

    if os.name == "nt":
        import winreg

        # An already open terminal or IDE may still have its old PATH.
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as key:
                cuda_path, _ = winreg.QueryValueEx(key, "CUDA_PATH_V11_2")
        except OSError:
            cuda_path = os.environ.get("CUDA_PATH_V11_2")
        if cuda_path:
            return Path(os.path.expandvars(cuda_path)) / "bin"
    return None


def configure_gpu():
    cuda_bin = cuda_bin_dir()
    if cuda_bin and cuda_bin.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(cuda_bin) != path_entries[0]:
            os.environ["PATH"] = os.pathsep.join([str(cuda_bin), *path_entries])
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "TensorFlow cannot see the NVIDIA GPU. Check that the NVIDIA driver, "
            "CUDA 11.2, and cuDNN 8 are installed and that TensorFlow can find their DLLs."
        )
    device = tf.config.experimental.get_device_details(gpus[0])
    print(f"TensorFlow {tf.__version__} | GPU: {device['device_name']}", flush=True)
