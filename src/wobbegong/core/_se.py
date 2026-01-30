import os

from singlecellexperiment import SingleCellExperiment
from summarizedexperiment import SummarizedExperiment

from .utils import _get_type_string, _sanitize_array, _write_json, dump_list_of_vectors, get_byte_order
from .wobbegongify import wobbegongify

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@wobbegongify.register
def wobbegongify_se(x: SummarizedExperiment, path: str) -> None:
    """Convert a SummarizedExperiment object to the wobbegong format.

    Args:
        x:
            SummarizedExperiment object to save.

        path:
            Path to store object.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    _row_data = x.get_row_data()
    if _row_data is not None and _row_data.shape[1] > 0:
        wobbegongify(_row_data, os.path.join(path, "row_data"))

    _col_data = x.get_column_data()
    if _col_data is not None and _col_data.shape[1] > 0:
        wobbegongify(_col_data, os.path.join(path, "column_data"))

    assay_names = x.get_assay_names()
    valid_assays = []

    assays_dir = os.path.join(path, "assays")
    if not os.path.exists(assays_dir):
        os.makedirs(assays_dir)

    for i, name in enumerate(assay_names):
        mat = x.get_assay(name)

        if len(mat.shape) != 2:
            continue

        wobbegongify(mat, os.path.join(assays_dir, str(len(valid_assays))))
        valid_assays.append(name)

    summary = {
        "object": "summarized_experiment",
        "row_count": x.shape[0],
        "column_count": x.shape[1],
        "has_row_data": _row_data is not None and _row_data.shape[1] > 0,
        "has_column_data": _col_data is not None and _col_data.shape[1] > 0,
        "assay_names": valid_assays,
    }

    if isinstance(x, SingleCellExperiment):
        summary["object"] = "single_cell_experiment"
        _handle_sce_parts(x, path, summary)

    _write_json(summary, os.path.join(path, "summary.json"))


def _handle_sce_parts(x: SingleCellExperiment, path: str, summary: dict) -> None:
    """Handle SingleCellExperiment specific parts (reduced dims, alt exps).

    Args:
        x:
            SingleCellExperiment object.

        path:
            Path to store object.

        summary:
            Summary dictionary to update.
    """
    rd_names = x.get_reduced_dimension_names()
    summary["reduced_dimension_names"] = rd_names

    rd_dir = os.path.join(path, "reduced_dimensions")
    if not os.path.exists(rd_dir):
        os.makedirs(rd_dir)

    for i, name in enumerate(rd_names):
        rd = x.get_reduced_dimension(name)
        curr_dir = os.path.join(rd_dir, str(i))
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        columns = []
        types = []
        col_names = [str(k) for k in range(rd.shape[1])]

        for c in range(rd.shape[1]):
            col = rd[:, c]
            t_str = _get_type_string(col.dtype)
            columns.append(_sanitize_array(col, t_str))
            types.append(t_str)

        content_path = os.path.join(curr_dir, "content")
        bytes_list = dump_list_of_vectors(columns, types, content_path)

        rd_summ = {
            "object": "data_frame",
            "byte_order": get_byte_order(),
            "row_count": rd.shape[0],
            "columns": {"names": col_names, "types": types, "bytes": bytes_list},
        }
        _write_json(rd_summ, os.path.join(curr_dir, "summary.json"))

    ae_names = x.get_alternative_experiment_names()
    summary["alternative_experiment_names"] = ae_names

    ae_dir = os.path.join(path, "alternative_experiments")
    if not os.path.exists(ae_dir):
        os.makedirs(ae_dir)

    for i, name in enumerate(ae_names):
        ae = x.get_alternative_experiment(name)
        wobbegongify(ae, os.path.join(ae_dir, str(i)))


@wobbegongify.register
def _(x: SingleCellExperiment, path: str) -> None:
    """Convert SingleCellExperiment to wobbegong format."""
    return wobbegongify.registry[SummarizedExperiment](x, path)
