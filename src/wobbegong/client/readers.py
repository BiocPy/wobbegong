import os

import numpy as np

from ._base import Accessor, HttpAccessor, LocalAccessor
from .utils import _parse_bytes

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class WobbegongBase:
    def __init__(self, accessor: Accessor, summary=None):
        self.accessor = accessor
        self.summary = summary if summary else self.accessor.read_json("summary.json")


class WobbegongDataFrame(WobbegongBase):
    def __init__(self, accessor, summary=None):
        super().__init__(accessor, summary)
        self.colnames = self.summary["columns"]["names"]
        self.coltypes = self.summary["columns"]["types"]
        self.colbytes = self.summary["columns"]["bytes"]
        self.shape = (self.summary["row_count"], len(self.colnames))

    def get_column(self, index_or_name):
        if isinstance(index_or_name, str):
            try:
                idx = self.colnames.index(index_or_name)
            except ValueError:
                raise KeyError(f"Column '{index_or_name}' not found")
        else:
            idx = index_or_name

        if idx < 0 or idx >= len(self.colnames):
            raise IndexError("Column index out of bounds")

        start = sum(self.colbytes[:idx])
        length = self.colbytes[idx]

        raw = self.accessor.read_bytes("content", start, length)
        return _parse_bytes(raw, self.coltypes[idx])

    def get_row_names(self):
        if not self.summary.get("has_row_names", False):
            return None

        start = sum(self.colbytes[:-1])
        length = self.colbytes[-1]
        raw = self.accessor.read_bytes("content", start, length)
        return _parse_bytes(raw, "string")


class WobbegongMatrix(WobbegongBase):
    def __init__(self, accessor, summary=None):
        super().__init__(accessor, summary)
        self.shape = (self.summary["row_count"], self.summary["column_count"])
        self.format = self.summary.get("format", "dense")
        self.dtype = self.summary["type"]
        self._stats_cache = {}

    def get_row(self, row_idx):
        if row_idx < 0 or row_idx >= self.shape[0]:
            raise IndexError(f"Row index {row_idx} out of bounds")

        if self.format == "dense":
            return self._get_dense_row(row_idx)
        else:
            return self._get_sparse_row(row_idx)

    def _get_dense_row(self, row_idx):
        row_bytes = self.summary["row_bytes"]
        start = sum(row_bytes[:row_idx])
        length = row_bytes[row_idx]

        raw = self.accessor.read_bytes("content", start, length)
        return _parse_bytes(raw, self.dtype)

    def _get_sparse_row(self, row_idx):
        v_bytes = self.summary["row_bytes"]["value"]
        i_bytes = self.summary["row_bytes"]["index"]

        prev_vals = sum(v_bytes[:row_idx])
        prev_idxs = sum(i_bytes[:row_idx])
        start = prev_vals + prev_idxs

        raw_vals = self.accessor.read_bytes("content", start, v_bytes[row_idx])
        values = _parse_bytes(raw_vals, self.dtype)

        start_idx = start + v_bytes[row_idx]
        raw_idxs = self.accessor.read_bytes("content", start_idx, i_bytes[row_idx])

        deltas = _parse_bytes(raw_idxs, "integer")
        indices = np.cumsum(deltas)

        out = np.zeros(self.shape[1], dtype=values.dtype)
        if len(indices) > 0:
            out[indices] = values
        return out

    def get_statistic(self, name):
        """Fetch a pre-calculated statistic (e.g. 'row_sum')"""
        stats_info = self.summary.get("statistics")
        if not stats_info or name not in stats_info["names"]:
            raise KeyError(f"Statistic '{name}' not available")

        idx = stats_info["names"].index(name)
        start = sum(stats_info["bytes"][:idx])
        length = stats_info["bytes"][idx]
        dtype = stats_info["types"][idx]

        raw = self.accessor.read_bytes("stats", start, length)
        return _parse_bytes(raw, dtype)


class WobbegongSummarizedExperiment(WobbegongBase):
    def __init__(self, accessor, summary=None):
        super().__init__(accessor, summary)
        self.assay_names = self.summary.get("assay_names", [])

    def get_row_data(self):
        if not self.summary.get("has_row_data"):
            return None
        return _load_wobbegong_dir(self.accessor, "row_data")

    def get_col_data(self):
        if not self.summary.get("has_column_data"):
            return None
        return _load_wobbegong_dir(self.accessor, "column_data")

    def get_assay(self, index_or_name):
        if isinstance(index_or_name, str):
            try:
                idx = self.assay_names.index(index_or_name)
            except ValueError:
                raise KeyError(f"Assay '{index_or_name}' not found")
        else:
            idx = index_or_name

        if idx < 0 or idx >= len(self.assay_names):
            raise IndexError("Assay index out of bounds")

        return _load_wobbegong_dir(self.accessor, f"assays/{idx}")


class WobbegongSingleCellExperiment(WobbegongSummarizedExperiment):
    def get_reduced_dim(self, name):
        names = self.summary.get("reduced_dimension_names", [])
        if name not in names:
            raise KeyError(f"Reduced dim '{name}' not found")

        idx = names.index(name)
        return _load_wobbegong_dir(self.accessor, f"reduced_dimensions/{idx}")


def _load_wobbegong_dir(parent_accessor, relative_path):
    """Creates a new accessor for a subdirectory (e.g., 'assays/0')

    and returns the appropriate Wobbegong object.
    """
    if isinstance(parent_accessor, LocalAccessor):
        new_path = os.path.join(parent_accessor.base_path, relative_path)
        accessor = LocalAccessor(new_path)
    else:
        new_url = f"{parent_accessor.base_url}/{relative_path}"
        accessor = HttpAccessor(new_url)

    summary = accessor.read_json("summary.json")
    obj_type = summary["object"]

    if obj_type == "data_frame":
        return WobbegongDataFrame(accessor, summary)
    elif obj_type == "matrix":
        return WobbegongMatrix(accessor, summary)
    elif obj_type == "summarized_experiment":
        return WobbegongSummarizedExperiment(accessor, summary)
    elif obj_type == "single_cell_experiment":
        return WobbegongSingleCellExperiment(accessor, summary)
    else:
        raise ValueError(f"Unknown object type: {obj_type}")
