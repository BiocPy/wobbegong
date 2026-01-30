import numpy as np
import pytest
from biocframe import BiocFrame
from scipy import sparse
from singlecellexperiment import SingleCellExperiment

from wobbegong import load, wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_client_local_dataframe(tmp_path):
    df = BiocFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]}, row_names=["r1", "r2", "r3"])

    path = str(tmp_path / "test_df")
    wobbegongify(df, path)

    wdf = load(path)

    assert wdf.colnames == ["A", "B"]

    col_a = wdf.get_column("A")
    np.testing.assert_array_equal(col_a, np.array([1, 2, 3], dtype=np.int32))

    col_b = wdf.get_column(1)  # Index access
    assert list(col_b) == ["x", "y", "z"]

    rows = wdf.get_row_names()
    assert list(rows) == ["r1", "r2", "r3"]

    with pytest.raises(KeyError):
        wdf.get_column("MISSING")


def test_client_local_matrix_dense(tmp_path):
    mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    path = str(tmp_path / "test_mat_dense")
    wobbegongify(mat, path)

    wmat = load(path)

    assert wmat.shape == (2, 3)
    assert wmat.format == "dense"

    r0 = wmat.get_row(0)
    np.testing.assert_array_equal(r0, [1, 2, 3])

    r1 = wmat.get_row(1)
    np.testing.assert_array_equal(r1, [4, 5, 6])

    r_sum = wmat.get_statistic("row_sum")
    np.testing.assert_array_equal(r_sum, [6, 15])


def test_client_local_matrix_sparse(tmp_path):
    mat = sparse.csr_matrix([[1.0, 0, 0, 2.0], [0, 0, 3.0, 0], [0, 0, 0, 0]])

    path = str(tmp_path / "test_mat_sparse")
    wobbegongify(mat, path)

    wmat = load(path)

    assert wmat.shape == (3, 4)
    assert wmat.format == "sparse"

    np.testing.assert_array_equal(wmat.get_row(0), [1.0, 0, 0, 2.0])
    np.testing.assert_array_equal(wmat.get_row(1), [0, 0, 3.0, 0])
    np.testing.assert_array_equal(wmat.get_row(2), [0, 0, 0, 0])


def test_client_local_sce(tmp_path):
    counts = np.random.randint(0, 10, (5, 3))
    pca = np.random.randn(3, 2)
    sce = SingleCellExperiment(assays={"counts": counts}, reduced_dims={"PCA": pca})

    path = str(tmp_path / "test_sce")
    wobbegongify(sce, path)

    wsce = load(path)

    wcounts = wsce.get_assay("counts")
    np.testing.assert_array_equal(wcounts.get_row(0), counts[0, :])

    wpca = wsce.get_reduced_dim("PCA")
    np.testing.assert_allclose(wpca.get_column(0), pca[:, 0])
