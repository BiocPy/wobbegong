import json
import os

import numpy as np
import pytest
from scipy import sparse

from wobbegong import wobbegongify
from wobbegong.client.utils import (
    read_double,
    read_integer,
    read_sparse_row_values,
    reconstruct_sparse_row,
)

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


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
        real_rnnz = mat.count_nonzero(axis=1)
        real_cnnz = mat.count_nonzero(axis=0)
    else:
        real_rnnz = np.count_nonzero(mat, axis=1)
        real_cnnz = np.count_nonzero(mat, axis=0)

    np.testing.assert_array_equal(rnnz, real_rnnz)
    np.testing.assert_array_equal(cnnz, real_cnnz)


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


def test_sparse_double_matrix(temp_dir):
    mat = sparse.random(20, 10, density=0.2, format="csr").astype(np.float64)

    if os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    wobbegongify(mat, temp_dir)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["format"] == "sparse"
    assert summary["type"] == "double"

    con_path = os.path.join(temp_dir, "content")
    v_bytes = summary["row_bytes"]["value"]
    i_bytes = summary["row_bytes"]["index"]

    total_lens = []
    for v, i in zip(v_bytes, i_bytes):
        total_lens.append(v)
        total_lens.append(i)

    offsets = [0] + list(np.cumsum(total_lens)[:-1])

    for r in [0, 10, 19]:
        v_len = v_bytes[r]
        i_len = i_bytes[r]
        start_pos = offsets[r * 2]

        vals, indices = read_sparse_row_values(con_path, start_pos, v_len, i_len, read_double)
        row_recon = reconstruct_sparse_row(vals, indices, 10, np.float64)
        row_orig = mat[r, :].toarray().flatten()

        np.testing.assert_allclose(row_recon, row_orig)

    check_stats(mat, os.path.join(temp_dir, "stats"), summary)
