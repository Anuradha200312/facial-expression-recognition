import io
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Add backend directory to sys.path so pytest can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_valid_image_bytes():
    """Generates a simple 100x100 RGB image byte stream."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_small_image_bytes():
    """Generates a 10x10 RGB image byte stream (below min resolution)."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_corrupt_bytes():
    return b"not an image binary payload"
