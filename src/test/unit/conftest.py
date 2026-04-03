import sys
from unittest.mock import MagicMock

_heavy_modules = [
    "prometheus_fastapi_instrumentator",
    "prometheus_client",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.dates",
]

for mod in _heavy_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()