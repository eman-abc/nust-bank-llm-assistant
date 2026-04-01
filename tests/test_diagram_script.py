from pathlib import Path
import shutil

from scripts.generate_architecture_diagram import generate_architecture_artifacts


class _FakeGraph:
    def draw_mermaid_png(self):
        return b"fake-png"

    def draw_mermaid(self):
        return "graph TD; A-->B;"


def test_generate_architecture_artifacts_writes_files():
    output_dir = Path.cwd() / "docs" / "test_architecture"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        artifacts = generate_architecture_artifacts(output_dir, graph=_FakeGraph())
        assert artifacts["png"].exists()
        assert artifacts["png"].read_bytes() == b"fake-png"
        assert artifacts["mermaid"].read_text(encoding="utf-8") == "graph TD; A-->B;"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
