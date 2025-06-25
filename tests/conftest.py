import os


def pytest_ignore_collect(path, config):
    # Skip this test during CI only
    return os.getenv("CI", "false").lower() == "true" and "test_app_main.py" in str(
        path
    )
