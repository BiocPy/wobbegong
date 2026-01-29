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


def read_boolean(path, start, length):
    data = read_and_decompress(path, start, length)
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.array([True if x == 1 else False if x == 0 else None for x in arr], dtype=object)

    return out


def read_string(path, start, length):
    data = read_and_decompress(path, start, length)
    full_str = data.decode("utf-8")

    # strings end with `\0`
    if full_str.endswith("\0"):
        full_str = full_str[:-1]

    values = full_str.split("\0")
    return values


def read_sparse_row_values(path, start, vlen, ilen, reader_func):
    vals = reader_func(path, start, vlen)

    # indices are delta encoded integers
    idx_bytes = read_and_decompress(path, start + vlen, ilen)
    deltas = np.frombuffer(idx_bytes, dtype=np.int32)
    indices = np.cumsum(deltas)

    return vals, indices


def reconstruct_sparse_row(vals, indices, ncols, dtype):
    out = np.zeros(ncols, dtype=dtype)
    out[indices] = vals
    return out
