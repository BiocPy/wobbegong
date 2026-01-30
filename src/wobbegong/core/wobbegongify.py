from functools import singledispatch
from typing import Any, Literal

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@singledispatch
def wobbegongify(x: Any, path: str, compression: Literal["lz4", "zlib"] = "zlib") -> None:
    """Convert an object to the wobbegong format.

    Args:
        x:
            Object to save to disk.

        path:
            Path to store object.

        compression:
            Compression method to use, either 'lz4' or 'zlib' (default).
    """
    raise NotImplementedError(f"No method for type: {type(x)}")
