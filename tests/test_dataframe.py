import os
import shutil
import pytest
import numpy as np
import json
from biocframe import BiocFrame
from wobbegong import wobbegongify
from wobbegong.client.utils import read_integer, read_double

@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "df_test"
    d.mkdir()
    return str(d)

def test_basic_dataframe(temp_dir):
    df = BiocFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [1.1, 2.2, 3.3, 4.4, 5.5],
    })

    wobbegongify(df, temp_dir)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["columns"]["names"] == ["A", "B"]
    assert summary["columns"]["types"] == ["integer", "double"]
    assert summary["row_count"] == 5
    assert not summary["has_row_names"]

    con_path = os.path.join(temp_dir, "content")
    bytes_lens = summary["columns"]["bytes"]
    ends = np.cumsum(bytes_lens)
    starts = [0] + list(ends[:-1])

    res_a = read_integer(con_path, starts[0], bytes_lens[0])
    np.testing.assert_array_equal(res_a, df.column("A"))

    res_b = read_double(con_path, starts[1], bytes_lens[1])
    np.testing.assert_array_almost_equal(res_b, df.column("B"))
