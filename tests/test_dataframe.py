import json
import os
import shutil

import numpy as np
import pytest
from biocframe import BiocFrame

from wobbegong import wobbegongify
from wobbegong.client.utils import read_boolean, read_double, read_integer, read_string

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "df_test"
    d.mkdir()
    return str(d)


@pytest.mark.parametrize("compression", ["zlib", "lz4"])
def test_basic_dataframe(temp_dir, compression):
    df = BiocFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": ["akari", "ai", "alice", "alicia", "athena"],
            "D": [True, False, True, False, True],
        }
    )

    wobbegongify(df, temp_dir, compression)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["columns"]["names"] == ["A", "B", "C", "D"]
    assert summary["columns"]["types"] == ["integer", "double", "string", "boolean"]
    assert summary["row_count"] == 5
    assert not summary["has_row_names"]
    assert summary["compression"] == compression

    con_path = os.path.join(temp_dir, "content")
    bytes_lens = summary["columns"]["bytes"]
    ends = np.cumsum(bytes_lens)
    starts = [0] + list(ends[:-1])

    res_a = read_integer(con_path, starts[0], bytes_lens[0], compression)
    np.testing.assert_array_equal(res_a, df.column("A"))

    res_b = read_double(con_path, starts[1], bytes_lens[1], compression)
    np.testing.assert_array_almost_equal(res_b, df.column("B"))

    res_c = read_string(con_path, starts[2], bytes_lens[2], compression)
    assert res_c == df.column("C")

    res_d = read_boolean(con_path, starts[3], bytes_lens[3], compression)
    np.testing.assert_array_equal(res_d, df.column("D"))


def test_dataframe_rownames(temp_dir):
    df = BiocFrame({"foo": [1.1, 2.2]}, row_names=["row1", "row2"])

    wobbegongify(df, temp_dir)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["has_row_names"] is True
    con_path = os.path.join(temp_dir, "content")
    bytes_lens = summary["columns"]["bytes"]

    start = sum(bytes_lens[:-1])
    length = bytes_lens[-1]

    res_rownames = read_string(con_path, start, length)
    assert res_rownames == list(df.row_names)
