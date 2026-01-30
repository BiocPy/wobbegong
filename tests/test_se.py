import json
import os
import shutil

import numpy as np
import pytest
from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

from wobbegong import wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "se_test"
    d.mkdir()
    return str(d)


def test_summarized_experiment(temp_dir):
    counts = np.random.randint(0, 10, (10, 5)).astype(np.int32)
    row_data = BiocFrame({"gene_id": [f"G{i}" for i in range(10)]})
    col_data = BiocFrame({"sample_id": [f"S{i}" for i in range(5)]})

    se = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    wobbegongify(se, temp_dir)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["object"] == "summarized_experiment"
    assert summary["row_count"] == 10
    assert summary["column_count"] == 5
    assert summary["has_row_data"] is True
    assert summary["has_column_data"] is True
    assert summary["assay_names"] == ["counts"]

    assert os.path.exists(os.path.join(temp_dir, "row_data", "summary.json"))
    assert os.path.exists(os.path.join(temp_dir, "column_data", "summary.json"))
    assert os.path.exists(os.path.join(temp_dir, "assays", "0", "summary.json"))

    rd_summ = json.load(open(os.path.join(temp_dir, "row_data", "summary.json")))
    assert rd_summ["columns"]["names"] == ["gene_id"]
