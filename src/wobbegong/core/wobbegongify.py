from functools import singledispatch

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@singledispatch
def wobbegongify(x, path: str):
    """Convert an object to the wobbegong format.

    Args:
        x:
            Object to save to disk.

        path:
            Path to store object.
    """
    raise NotImplementedError(f"No method for type: {type(x)}")
