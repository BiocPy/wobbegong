import os

import numpy as np
from delayedarray import DelayedArray
from scipy import sparse

from .utils import _get_type_string, _sanitize_array, _write_json, dump_list_of_vectors, dump_matrix, get_byte_order
from .wobbegongify import wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def wobbegongify_matrix(x, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    con_path = os.path.join(path, "content")
    type_str = _get_type_string(x.dtype)

    summary = {
        "object": "matrix",
        "byte_order": get_byte_order(),
        "row_count": x.shape[0],
        "column_count": x.shape[1],
        "type": type_str,
    }

    res = dump_matrix(x, con_path, type_str)

    if not res["is_sparse"]:
        summary["format"] = "dense"
        summary["row_bytes"] = res["row_bytes_dense"]

    stats_path = os.path.join(path, "stats")
    stat_names = ["row_sum", "column_sum", "row_nonzero", "column_nonzero"]
    stat_types = [type_str, type_str, "integer", "integer"]

    stat_arrays = [
        _sanitize_array(res["row_sums"], stat_types[0]),
        _sanitize_array(res["col_sums"], stat_types[1]),
        _sanitize_array(res["row_nonzero"], stat_types[2]),
        _sanitize_array(res["col_nonzero"], stat_types[3]),
    ]

    stat_bytes = dump_list_of_vectors(stat_arrays, stat_types, stats_path)
    summary["statistics"] = {"names": stat_names, "types": stat_types, "bytes": stat_bytes}
    _write_json(summary, os.path.join(path, "summary.json"))


@wobbegongify.register(np.ndarray)
def _(x, path):
    wobbegongify_matrix(x, path)


# @wobbegongify.register(sparse.spmatrix)
# def _(x, path):
#     wobbegongify_matrix(x, path)


# @wobbegongify.register(DelayedArray)
# def _(x, path):
#     wobbegongify_matrix(x, path)
