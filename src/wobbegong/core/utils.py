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
