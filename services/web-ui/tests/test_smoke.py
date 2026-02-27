"""Smoke tests for the Web-UI service."""


def test_import_app():
    """Verify the Streamlit app module can be imported without errors."""
    import app  # noqa: F401
