"""FedMed Flower App package."""

from . import client_app as client_app  # re-export for Flower discovery
from . import server_app as server_app
from . import task as task

__all__ = [
    "task",
    "client_app",
    "server_app",
]

__version__ = "0.0.9"
