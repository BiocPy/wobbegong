import json
import sys
from typing import Literal

import delayedarray
import mattress
import numpy as np

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def _write_json(data: dict, path: str) -> None:
    """Write dictionary to a JSON file.

    Args:
        data:
            Data to write.

        path:
            Path to the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _get_type_string(dtype: np.dtype) -> str:
    """Get Wobbegong type string from numpy dtype.

    Args:
        dtype:
            Numpy dtype.

    Returns:
        Wobbegong type string ('integer', 'double', 'boolean', 'string').
    """
    if np.issubdtype(dtype, np.integer):
        return "integer"

    if np.issubdtype(dtype, np.floating):
        return "double"

    if np.issubdtype(dtype, np.bool_):
        return "boolean"

    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
        return "string"

    return "string"


def _sanitize_array(x: np.ndarray, type_str: str) -> np.ndarray:
    """Sanitize array to ensure correct type for Wobbegong.

    Args:
        x:
            Input array.

        type_str:
            Expected Wobbegong type string.

    Returns:
        Sanitized array.
    """
    if type_str == "integer":
        return x.astype(np.int32, copy=False)
    elif type_str == "double":
        return x.astype(np.float64, copy=False)
    return x


def get_byte_order() -> str:
    """Return byte order of the machine.

    Returns:
        'little_endian' or 'big_endian'.
    """
    return "little_endian" if sys.byteorder == "little" else "big_endian"


def compress_and_write(f, data_bytes: bytes, compression: Literal["lz4", "zlib"] = "zlib") -> int:
    """Use zlib to compress and write data to file.

    Args:
        f:
            File writer object.

        data_bytes:
            Data to write (in bytes).

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Length of the compressed data written.
    """
    if compression == "lz4":
        import lz4.block

        compressed = lz4.block.compress(data_bytes, store_size=False)
    else:
        import zlib

        compressed = zlib.compress(data_bytes)

    f.write(compressed)
    return len(compressed)


def dump_list_of_vectors(
    columns: list, types: list[str], filepath: str, compression: Literal["lz4", "zlib"] = "zlib"
) -> list[int]:
    """Write a list of vectors to disk.

    Args:
        columns:
            List of column vectors (arrays) to write.

        types:
            List identifying the type for each column.

        filepath:
            Path to write the data.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        List of compressed sizes for each column.
    """
    with open(filepath, "wb") as f:
        pass

    sizes = []
    with open(filepath, "ab") as f:
        for col, type_str in zip(columns, types):
            if type_str == "string":
                if isinstance(col, np.ndarray):
                    col = col.tolist()
                joined = "\0".join([str(x) for x in col]) + "\0"
                raw_bytes = joined.encode("utf-8")

            elif type_str == "boolean":
                raw_bytes = col.astype(np.uint8).tobytes()

            else:
                raw_bytes = col.tobytes()

            sizes.append(compress_and_write(f, raw_bytes, compression=compression))

    return sizes


def dump_matrix(
    x,
    filepath: str,
    type_str: str,
    chunk_size: int = 10000,
    num_threads: int = 1,
    compression: Literal["lz4", "zlib"] = "zlib",
) -> dict:
    """Uses mattress to initialize the matrix and iterate over rows.

    Calculates stats and writes compressed rows.

    Args:
        x:
            Matrix object. Refer to the mattress package for all
            supported matrix input types.

        filepath:
            Path to write.

        type_str:
            Wobbegong type string of the matrix data.

        chunk_size:
            Number of rows to read per chunk.
            Defaults to 10000.

        num_threads:
            Number of threads for stats calculation.
            Defaults to 1.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).

    Returns:
        Dictionary containing matrix statistics and file byte offsets.
    """
    ptr = mattress.initialize(x)
    is_sparse = ptr.sparse()
    nrows = ptr.nrow()
    ncols = ptr.ncol()

    row_nnz = np.zeros(nrows, dtype=np.int32)
    col_nnz = np.zeros(ncols, dtype=np.int32)

    row_bytes_dense = []
    row_bytes_val = []
    row_bytes_idx = []

    with open(filepath, "wb") as f:
        pass

    with open(filepath, "ab") as f:
        if is_sparse:
            tptr = delayedarray.Transpose(ptr, (1, 0))

            for start_row in range(0, nrows, chunk_size):
                end_row = min(start_row + chunk_size, nrows)

                sa = delayedarray.extract_sparse_array(tptr, (range(ncols), range(start_row, end_row)))

                for idx, row_content in enumerate(sa.contents):
                    global_row_idx = start_row + idx

                    if row_content is None:
                        data = np.array([], dtype=np.float64)
                        indices = np.array([], dtype=np.int32)
                    else:
                        indices, data = row_content
                        data = _sanitize_array(data, type_str)

                    count = len(data)
                    row_nnz[global_row_idx] = count
                    if count > 0:
                        col_nnz[indices] += 1

                    vb = compress_and_write(f, data.tobytes(), compression=compression)
                    row_bytes_val.append(vb)

                    if count > 0:
                        deltas = np.diff(indices, prepend=0)
                        deltas[0] = indices[0]
                        ib = compress_and_write(f, deltas.astype(np.int32).tobytes(), compression=compression)
                    else:
                        ib = compress_and_write(f, b"", compression=compression)
                    row_bytes_idx.append(ib)

        else:
            for start_row in range(0, nrows, chunk_size):
                end_row = min(start_row + chunk_size, nrows)

                chunk = delayedarray.extract_dense_array(ptr, (range(start_row, end_row), range(ncols)))
                chunk = _sanitize_array(chunk, type_str)

                nz_mask = chunk != 0
                row_nnz[start_row:end_row] = np.sum(nz_mask, axis=1)
                col_nnz += np.sum(nz_mask, axis=0)

                for i in range(chunk.shape[0]):
                    b = compress_and_write(f, chunk[i, :].tobytes(), compression=compression)
                    row_bytes_dense.append(b)

    r_sums = ptr.row_sums(num_threads=num_threads)
    c_sums = ptr.column_sums(num_threads=num_threads)

    return {
        "is_sparse": is_sparse,
        "row_sums": r_sums,
        "col_sums": c_sums,
        "row_nonzero": row_nnz,
        "col_nonzero": col_nnz,
        "row_bytes_dense": row_bytes_dense,
        "row_bytes_val": row_bytes_val,
        "row_bytes_idx": row_bytes_idx,
    }
