import json
import os
from abc import ABC, abstractmethod

import requests

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class Accessor(ABC):
    @abstractmethod
    def read_json(self, relative_path):
        pass

    @abstractmethod
    def read_bytes(self, relative_path, start, length) -> bytes:
        pass

    def close(self):
        pass


class LocalAccessor(Accessor):
    def __init__(self, base_path):
        self.base_path = base_path

    def read_json(self, relative_path):
        with open(os.path.join(self.base_path, relative_path), "r") as f:
            return json.load(f)

    def read_bytes(self, relative_path, start, length):
        path = os.path.join(self.base_path, relative_path)
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(length)


class HttpAccessor(Accessor):
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def read_json(self, relative_path):
        url = f"{self.base_url}/{relative_path}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def read_bytes(self, relative_path, start, length):
        url = f"{self.base_url}/{relative_path}"
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content
