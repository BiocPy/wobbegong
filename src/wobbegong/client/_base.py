import json
import os
from abc import ABC, abstractmethod

import requests

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class Accessor(ABC):
    """Abstract base class for accessing data."""

    @abstractmethod
    def read_json(self, relative_path: str) -> dict:
        """Read a JSON file.

        Args:
            relative_path:
                Path to the JSON file relative to the base.

        Returns:
            The parsed JSON content as a dictionary.
        """
        pass

    @abstractmethod
    def read_bytes(self, relative_path: str, start: int, length: int) -> bytes:
        """Read a chunk of bytes from a file.

        Args:
            relative_path:
                Path to the file relative to the base.

            start:
                Start position (offset) in bytes.

            length:
                Number of bytes to read.

        Returns:
            The raw bytes read from the file.
        """
        pass

    def close(self) -> None:
        """Close any open resources."""
        pass


class LocalAccessor(Accessor):
    """Accessor for local file system."""

    def __init__(self, base_path: str):
        """Initialize the local accessor.

        Args:
            base_path:
                Base directory path.
        """
        self.base_path = base_path

    def read_json(self, relative_path: str) -> dict:
        """Read a JSON file from the local filesystem.

        Args:
            relative_path:
                Path relative to the base directory.

        Returns:
            The parsed JSON content.
        """
        with open(os.path.join(self.base_path, relative_path), "r") as f:
            return json.load(f)

    def read_bytes(self, relative_path: str, start: int, length: int) -> bytes:
        """Read bytes from a local file.

        Args:
            relative_path:
                Path relative to the base directory.

            start:
                Start offset in bytes.

            length:
                Number of bytes to read.

        Returns:
            The bytes read.
        """
        path = os.path.join(self.base_path, relative_path)
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(length)


class HttpAccessor(Accessor):
    """Accessor for HTTP resources."""

    def __init__(self, base_url: str):
        """Initialize the HTTP accessor.

        Args:
            base_url:
                Base URL for HTTP requests.
        """
        self.base_url = base_url.rstrip("/")

    def read_json(self, relative_path: str) -> dict:
        """Read a JSON file relative to the base URL.

        Args:
            relative_path:
                Path relative to the base URL.

        Returns:
            The parsed JSON content.
        """
        url = f"{self.base_url}/{relative_path}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def read_bytes(self, relative_path: str, start: int, length: int) -> bytes:
        """Read specific bytes from a file relative to the base URL.

        Args:
            relative_path:
                Path relative to the base URL.

            start:
                Start offset in bytes.

            length:
                Number of bytes to read.

        Returns:
            The bytes read.
        """
        url = f"{self.base_url}/{relative_path}"
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content
