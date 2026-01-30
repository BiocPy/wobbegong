from functools import singledispatch
from typing import Any

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@singledispatch
def wobbegongify(x: Any, path: str) -> None:
    """Convert an object to the wobbegong format.

    Args:
        x:
            Object to save to disk.

        path:
            Path to store object.
    """
    raise NotImplementedError(f"No method for type: {type(x)}")
