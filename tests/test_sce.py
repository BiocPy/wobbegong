import json
import os
import shutil

import numpy as np
import pytest
from biocframe import BiocFrame
from singlecellexperiment import SingleCellExperiment

from wobbegong import wobbegongify
from wobbegong.client.utils import read_double

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "sce_test"
    d.mkdir()
    return str(d)


@pytest.mark.parametrize("compression", ["zlib", "lz4"])
def test_sce_red_dims(temp_dir, compression):
    counts = np.zeros((10, 5), dtype=np.int32)
    pca = np.random.randn(5, 2)
    tsne = np.random.randn(5, 2)

    sce = SingleCellExperiment(assays={"counts": counts}, reduced_dims={"PCA": pca, "TSNE": tsne})

    wobbegongify(sce, temp_dir, compression)

    with open(os.path.join(temp_dir, "summary.json")) as f:
        summary = json.load(f)

    assert summary["object"] == "single_cell_experiment"
    assert summary["compression"] == compression
    assert "PCA" in summary["reduced_dimension_names"]
    assert "TSNE" in summary["reduced_dimension_names"]

    pca_dir = os.path.join(temp_dir, "reduced_dimensions", "0")
    with open(os.path.join(pca_dir, "summary.json")) as f:
        pca_summ = json.load(f)

    assert pca_summ["row_count"] == 5

    assert pca_summ["object"] == "data_frame"
    assert pca_summ["columns"]["types"][0] == "double"

    con_path = os.path.join(pca_dir, "content")
    col1_bytes = pca_summ["columns"]["bytes"][0]

    res = read_double(con_path, 0, col1_bytes, compression)
    np.testing.assert_allclose(res, pca[:, 0])


@pytest.mark.parametrize("compression", ["zlib", "lz4"])
def test_sce_alt_exps(temp_dir, compression):
    counts = np.zeros((10, 5), dtype=np.int32)
    alt_counts = np.ones((3, 5), dtype=np.int32)
    alt = SingleCellExperiment(assays={"counts": alt_counts})

    sce = SingleCellExperiment(assays={"counts": counts}, alternative_experiments={"Spikes": alt})

    wobbegongify(sce, temp_dir, compression)

    alt_dir = os.path.join(temp_dir, "alternative_experiments", "0")
    assert os.path.exists(os.path.join(alt_dir, "summary.json"))

    with open(os.path.join(alt_dir, "summary.json")) as f:
        alt_summ = json.load(f)

    assert alt_summ["row_count"] == 3
    assert alt_summ["column_count"] == 5
