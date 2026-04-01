from __future__ import annotations

import csv
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cleaned = [cell.strip() for cell in row if cell and cell.strip()]
            if cleaned:
                rows.append(", ".join(cleaned))
    return "\n".join(rows)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page and page.strip())


def parse_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_text(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported document type: {suffix}")
