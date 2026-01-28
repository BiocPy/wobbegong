import zlib

import numpy as np

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def read_and_decompress(path, start, length):
    with open(path, "rb") as f:
        f.seek(start)
        compressed = f.read(length)

    return zlib.decompress(compressed)


def read_integer(path, start, length):
    data = read_and_decompress(path, start, length)
    return np.frombuffer(data, dtype=np.int32)


def read_double(path, start, length):
    data = read_and_decompress(path, start, length)

    return np.frombuffer(data, dtype=np.float64)
