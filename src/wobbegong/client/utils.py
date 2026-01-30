import zlib

import numpy as np

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def read_chunk(path, start, length):
    with open(path, "rb") as f:
        f.seek(start)
        compressed = f.read(length)
    return compressed


def read_integer(path, start, length):
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "integer")


def read_double(path, start, length):
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "double")


def read_boolean(path, start, length):
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "boolean")


def read_string(path, start, length):
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "string")


def read_sparse_row_values(path, start, vlen, ilen, reader_func):
    vals = reader_func(path, start, vlen)

    # indices are delta encoded integers
    idx_bytes_raw = read_chunk(path, start + vlen, ilen)
    idx_bytes = zlib.decompress(idx_bytes_raw)
    deltas = np.frombuffer(idx_bytes, dtype=np.int32)
    indices = np.cumsum(deltas)

    return vals, indices


def reconstruct_sparse_row(vals, indices, ncols, dtype):
    out = np.zeros(ncols, dtype=dtype)
    out[indices] = vals
    return out


def _parse_bytes(raw_bytes, dtype_str):
    decompressed = zlib.decompress(raw_bytes)

    if dtype_str == "integer":
        return np.frombuffer(decompressed, dtype=np.int32)
    elif dtype_str == "double":
        return np.frombuffer(decompressed, dtype=np.float64)
    elif dtype_str == "boolean":
        raw = np.frombuffer(decompressed, dtype=np.uint8)
        return np.array([True if x == 1 else False if x == 0 else None for x in raw], dtype=object)
    elif dtype_str == "string":
        text = decompressed.decode("utf-8")
        if text.endswith("\0"):
            text = text[:-1]
        # return np.array(text.split("\0"), dtype=object)
        return text.split("\0")
    else:
        raise ValueError(f"Unknown type: {dtype_str}")
