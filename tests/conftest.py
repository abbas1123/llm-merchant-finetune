import os

import pytest

LOCAL_ENABLED = os.environ.get("MERCHANT_LOCAL_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if LOCAL_ENABLED:
        return
    skip_local = pytest.mark.skip(reason="needs model weights; set MERCHANT_LOCAL_TESTS=1 to run")
    for item in items:
        if "local" in item.keywords:
            item.add_marker(skip_local)
