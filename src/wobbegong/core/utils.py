import json
import sys
import zlib

import delayedarray
import mattress
import numpy as np

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def _write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _get_type_string(dtype):
    if np.issubdtype(dtype, np.integer):
        return "integer"

    if np.issubdtype(dtype, np.floating):
        return "double"

    if np.issubdtype(dtype, np.bool_):
        return "boolean"

    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
        return "string"

    return "string"


def _sanitize_array(x, type_str):
    if type_str == "integer":
        return x.astype(np.int32, copy=False)
    elif type_str == "double":
        return x.astype(np.float64, copy=False)
    return x


def get_byte_order():
    """Return byte order."""
    return "little_endian" if sys.byteorder == "little" else "big_endian"


def compress_and_write(f, data_bytes) -> int:
    """Use zlib to compress and write data to file.

    Args:
        f:
            File writer.

        data_bytes:
            Data to write (in bytes).

    Returns:
        Length of the data_bytes after compression."""
    compressed = zlib.compress(data_bytes)
    f.write(compressed)
    return len(compressed)


def dump_list_of_vectors(columns, types, filepath):
    """Write a list of vectors to disk.

    Args:
        columns:
            Column vectors to write.

        types:
            List specifying type for each column.

        filepath:
            Path to write the data.
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

            sizes.append(compress_and_write(f, raw_bytes))

    return sizes


def coerce_data_chunk(data, type_str):
    if type_str == "integer":
        data = data.astype(np.int32, copy=False)
    elif type_str == "double":
        data = data.astype(np.float64, copy=False)

    return data


def dump_matrix(x, filepath, type_str, chunk_size=10000, num_threads=1):
    """Uses mattress to initialize the matrix and iterate over rows.

    Calculates stats and writes compressed rows.

    Args:
        x:
            Matrix object. Refer to the mattress package for all
            supported matrix input types.

        filepath:
            Path to write.

        type_str:
            NumPy dtype of the matrix.

        chunk_size:
            Number of rows to read.
            Defaults to 10000

        num_threads:
            Number of threads.
            Default to 1.
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
        if not is_sparse:
            for start_row in range(0, nrows, chunk_size):
                end_row = min(start_row + chunk_size, nrows)

                chunk = delayedarray.extract_dense_array(ptr, (range(start_row, end_row), range(ncols)))
                chunk = coerce_data_chunk(chunk, type_str)

                nz_mask = chunk != 0
                row_nnz[start_row:end_row] = np.sum(nz_mask, axis=1)
                col_nnz += np.sum(nz_mask, axis=0)

                for i in range(chunk.shape[0]):
                    b = compress_and_write(f, chunk[i, :].tobytes())
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
