import numpy as np
from typing import Literal

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def read_chunk(path: str, start: int, length: int) -> bytes:
    """Read a chunk of bytes from a file.

    Args:
        path:
            Path to the file.

        start:
            Start offset in bytes.

        length:
            Number of bytes to read.

    Returns:
        The bytes read.
    """
    with open(path, "rb") as f:
        f.seek(start)
        compressed = f.read(length)
    return compressed


def read_integer(path: str, start: int, length: int, compression: Literal["lz4", "zlib"] = "zlib") -> np.ndarray:
    """Read integer data from a file.

    Args:
        path:
            Path to the file.

        start:
            Start offset in bytes.

        length:
            Number of bytes to read.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Numpy array of integers.
    """
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "integer", compression=compression)


def read_double(path: str, start: int, length: int, compression: Literal["lz4", "zlib"] = "zlib") -> np.ndarray:
    """Read double (float) data from a file.

    Args:
        path:
            Path to the file.

        start:
            Start offset in bytes.

        length:
            Number of bytes to read.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Numpy array of floats.
    """
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "double", compression=compression)


def read_boolean(path: str, start: int, length: int, compression: Literal["lz4", "zlib"] = "zlib") -> np.ndarray:
    """Read boolean data from a file.

    Args:
        path:
            Path to the file.

        start:
            Start offset in bytes.

        length:
            Number of bytes to read.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Numpy array of booleans (or None/object).
    """
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "boolean", compression=compression)


def read_string(path: str, start: int, length: int, compression: Literal["lz4", "zlib"] = "zlib") -> list[str]:
    """Read string data from a file.

    Args:
        path:
            Path to the file.

        start:
            Start offset in bytes.

        length:
            Number of bytes to read.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        List of strings.
    """
    data = read_chunk(path, start, length)
    return _parse_bytes(data, "string", compression=compression)


def _decompress(data, compression):
    if compression == "lz4":
        import lz4.block

        return lz4.block.decompress(data, uncompressed_size=len(data) * 255)
    else:
        import zlib

        return zlib.decompress(data)


def read_sparse_row_values(
    path: str, start: int, vlen: int, ilen: int, reader_func: callable, compression: Literal["lz4", "zlib"] = "zlib"
):
    """Read sparse row values.

    Args:
        path:
            Path to the file.

        start:
            Start offset.

        vlen:
            Length of values in bytes.

        ilen:
            Length of indices in bytes.

        reader_func:
            Function to read the values (integer, double, etc.).

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Tuple of (values, indices).
    """
    vals = reader_func(path, start, vlen, compression)

    # indices are delta encoded integers
    idx_bytes_raw = read_chunk(path, start + vlen, ilen)
    idx_bytes = _decompress(idx_bytes_raw, compression)
    deltas = np.frombuffer(idx_bytes, dtype=np.int32)
    indices = np.cumsum(deltas)

    return vals, indices


def reconstruct_sparse_row(vals: np.ndarray, indices: np.ndarray, ncols: int, dtype: np.dtype):
    """Reconstruct a sparse row into a dense array.

    Args:
        vals:
            Values.

        indices:
            Column indices.

        ncols:
            Total number of columns.

        dtype:
            Data type of the array.

    Returns:
        Dense numpy array.
    """
    out = np.zeros(ncols, dtype=dtype)
    out[indices] = vals
    return out


def _parse_bytes(raw_bytes: bytes, dtype_str: str, compression: Literal["lz4", "zlib"] = "zlib") -> np.ndarray | list[str]:
    """Parse raw bytes into numpy array or list of strings.

    Args:
        raw_bytes:
            Compressed raw bytes.

        dtype_str:
            Type string ('integer', 'double', 'boolean', 'string').

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Parsed data.
    """
    decompressed = _decompress(raw_bytes, compression)

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


def _map_wobbegong_type_to_numpy(type_str: str) -> type | np.dtype:
    """Map Wobbegong type string to numpy type.

    Args:
        type_str:
            Wobbegong type string.

    Returns:
        Numpy type or dtype.
    """
    if type_str == "integer":
        return np.int32
    elif type_str == "double":
        return np.float64
    elif type_str == "boolean":
        return np.uint8
    else:
        return object
