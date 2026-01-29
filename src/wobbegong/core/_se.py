import os

from summarizedexperiment import SummarizedExperiment

from .utils import _write_json
from .wobbegongify import wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@wobbegongify.register
def wobbegongify_se(x: SummarizedExperiment, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    if x.row_data is not None and x.row_data.shape[1] > 0:
        wobbegongify(x.row_data, os.path.join(path, "row_data"))

    if x.col_data is not None and x.col_data.shape[1] > 0:
        wobbegongify(x.col_data, os.path.join(path, "column_data"))

    assay_names = list(x.assays.keys())
    valid_assays = []

    assays_dir = os.path.join(path, "assays")
    if not os.path.exists(assays_dir):
        os.makedirs(assays_dir)

    for i, name in enumerate(assay_names):
        mat = x.assays[name]
        if len(mat.shape) != 2:
            continue
        wobbegongify(mat, os.path.join(assays_dir, str(len(valid_assays))))
        valid_assays.append(name)

    summary = {
        "object": "summarized_experiment",
        "row_count": x.shape[0],
        "column_count": x.shape[1],
        "has_row_data": x.row_data is not None and x.row_data.shape[1] > 0,
        "has_column_data": x.col_data is not None and x.col_data.shape[1] > 0,
        "assay_names": valid_assays,
    }

    _write_json(summary, os.path.join(path, "summary.json"))
