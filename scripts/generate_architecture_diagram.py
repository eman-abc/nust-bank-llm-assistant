from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import get_settings


def generate_architecture_artifacts(output_dir: Path, graph=None) -> dict[str, Path]:
    if graph is None:
        from backend.orchestrator import bank_bot

        graph = bank_bot.get_graph()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "rag_architecture.png"
    mermaid_path = output_dir / "rag_architecture.mmd"

    png_bytes = graph.draw_mermaid_png()
    png_path.write_bytes(png_bytes)

    if hasattr(graph, "draw_mermaid"):
        mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")

    return {"png": png_path, "mermaid": mermaid_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mermaid graph artifacts for the RAG architecture.")
    parser.add_argument(
        "--output-dir",
        default=str(get_settings().architecture_artifacts_dir),
        help="Directory where the graph artifacts should be written.",
    )
    args = parser.parse_args()

    artifacts = generate_architecture_artifacts(Path(args.output_dir))
    print(f"Generated architecture diagram at {artifacts['png']}")


if __name__ == "__main__":
    main()
