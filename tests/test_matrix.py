import json
import os

import numpy as np
import pytest
from scipy import sparse

from wobbegong import wobbegongify
from wobbegong.client.utils import read_boolean, read_double, read_integer, read_sparse_row_values, reconstruct_sparse_row


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path / "mat_test")


def check_stats(mat, path, summary):
    stats_info = summary["statistics"]
    bytes_lens = stats_info["bytes"]
    ends = np.cumsum(bytes_lens)
    starts = [0] + list(ends[:-1])

    def get_stat(name, reader):
        idx = stats_info["names"].index(name)
        return reader(path, starts[idx], bytes_lens[idx])

    if summary["type"] == "double":
        rsums = get_stat("row_sum", read_double)
        csums = get_stat("column_sum", read_double)

        expected_r = np.asarray(np.sum(mat, axis=1)).flatten()
        expected_c = np.asarray(np.sum(mat, axis=0)).flatten()

        np.testing.assert_allclose(rsums, expected_r, rtol=1e-5)
        np.testing.assert_allclose(csums, expected_c, rtol=1e-5)

    rnnz = get_stat("row_nonzero", read_integer)
    cnnz = get_stat("column_nonzero", read_integer)

    if sparse.issparse(mat):
        real_rnnz = np.diff(mat.indptr)
    else:
        real_rnnz = np.count_nonzero(mat, axis=1)

    np.testing.assert_array_equal(rnnz, real_rnnz)


def test_dense_integer_matrix(temp_dir):
    mat = np.random.randint(0, 10, size=(10, 5)).astype(np.int32)
    os.makedirs(temp_dir, exist_ok=True)
    wobbegongify(mat, temp_dir)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["format"] == "dense"
    assert summary["type"] == "integer"

    con_path = os.path.join(temp_dir, "content")
    row_bytes = summary["row_bytes"]
    starts = [0] + list(np.cumsum(row_bytes)[:-1])

    for r in [0, 4, 9]:
        res = read_integer(con_path, starts[r], row_bytes[r])
        np.testing.assert_array_equal(res, mat[r, :])

    check_stats(mat, os.path.join(temp_dir, "stats"), summary)
