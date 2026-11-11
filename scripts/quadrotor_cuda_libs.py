#!/usr/bin/env python3
import glob
import os


def package_libs(package, pattern):
    module = __import__(package, fromlist=["__path__"])
    root = module.__path__[0]
    return root, sorted(glob.glob(os.path.join(root, "lib", pattern)))


def main():
    for package, pattern in (
        ("nvidia.cudnn", "libcudnn.so*"),
        ("nvidia.nccl", "libnccl.so*"),
    ):
        try:
            root, libs = package_libs(package, pattern)
            print(package, root)
            for lib in libs:
                print(" ", lib)
        except Exception as exc:
            print(package, "missing", repr(exc))

    nvml = []
    for directory in (
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ):
        nvml.extend(sorted(glob.glob(os.path.join(directory, "libnvidia-ml.so*"))))
    print("nvidia-ml")
    for lib in nvml:
        print(" ", lib)


if __name__ == "__main__":
    main()
