import os

import numpy as np
from biocframe import BiocFrame

from .utils import _get_type_string, _sanitize_array, _write_json, dump_list_of_vectors, get_byte_order
from .wobbegongify import wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@wobbegongify.register
def wobbegongify_frame(x: BiocFrame, path: str):
    """Convert a `BiocFrame` object to the wobbegong format.

    Stores data in columnar format to quickly retrieve an entire column.

    Args:
        x:
            BiocFrame to save to disk.

        path:
            Path to store object.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    summary = {
        "object": "data_frame",
        "byte_order": get_byte_order(),
        "row_count": x.shape[0],
        "has_row_names": x.row_names is not None,
    }

    columns = []
    types = []

    _colnames = x.get_column_names()
    for col_name in _colnames:
        col_data = x.get_column(col_name)
        if isinstance(col_data, list):
            col_data = np.array(col_data)

        t_str = _get_type_string(col_data.dtype)
        col_data = _sanitize_array(col_data, t_str)

        columns.append(col_data)
        types.append(t_str)

    _rownames = x.get_row_names()
    if _rownames is not None:
        columns.append(np.array(_rownames))
        types.append("string")

    con_path = os.path.join(path, "content")
    bytes_list = dump_list_of_vectors(columns, types, con_path)

    summary["columns"] = {"names": list(_colnames), "types": types[: len(_colnames)], "bytes": bytes_list}
    _write_json(summary, os.path.join(path, "summary.json"))
