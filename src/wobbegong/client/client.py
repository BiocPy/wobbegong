from urllib.parse import urlparse

from ._base import HttpAccessor, LocalAccessor
from .readers import (
    WobbegongDataFrame,
    WobbegongMatrix,
    WobbegongSingleCellExperiment,
    WobbegongSummarizedExperiment,
)

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def load(path_or_url):
    """Loads a Wobbegong dataset from a local path or URL.

    Args:
        path_or_url:
            Local path or URL.
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        accessor = HttpAccessor(path_or_url)
    else:
        accessor = LocalAccessor(path_or_url)

    summary = accessor.read_json("summary.json")
    obj_type = summary.get("object")

    if obj_type == "data_frame":
        return WobbegongDataFrame(accessor, summary)
    elif obj_type == "matrix":
        return WobbegongMatrix(accessor, summary)
    elif obj_type == "summarized_experiment":
        return WobbegongSummarizedExperiment(accessor, summary)
    elif obj_type == "single_cell_experiment":
        return WobbegongSingleCellExperiment(accessor, summary)
    else:
        raise ValueError(f"Unknown wobbegong object type in summary: {obj_type}")
