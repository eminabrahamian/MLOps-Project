"""
Unit test for simulating artifact download used in the ML pipeline.

Covers:
- Creating a temporary artifact directory
- Pretending to download model + pipeline artifacts
- Ensuring both files are present and accessible
"""

from pathlib import Path


class DummyArtifact:
    """Mock class to simulate .download() from a W&B or MLflow artifact."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def download(self) -> str:
        return str(self.path)


def test_artifact_download_structure(tmp_path):
    """
    Simulate model + preprocessing pipeline download
    and verify both expected files exist in the returned directory.
    """
    art_dir = tmp_path / "artifact"
    art_dir.mkdir()

    # Simulate model + pipeline files
    (art_dir / "model.pkl").write_text("fake-model")
    (art_dir / "preprocessing_pipeline.pkl").write_text("fake-pipeline")

    # Simulate artifact object behavior
    fake_artifact = DummyArtifact(art_dir)
    download_path = Path(fake_artifact.download())

    # Assertions
    assert (download_path / "model.pkl").is_file()
    assert (download_path / "preprocessing_pipeline.pkl").is_file()

    # Optional: confirm contents
    assert (download_path / "model.pkl").read_text() == "fake-model"
    assert (download_path / "preprocessing_pipeline.pkl").read_text() == "fake-pipeline"
