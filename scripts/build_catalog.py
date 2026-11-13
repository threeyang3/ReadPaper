#!/usr/bin/env python3
"""Build a deterministic, read-only catalog for PaperFlow and Markdown clients."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _safe_relative(value: str, base: Path) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"unsafe relative archive path: {value!r}")
    candidate = (base / value).resolve()
    candidate.relative_to(base.resolve())
    return candidate.relative_to(ROOT.resolve()).as_posix()


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_catalog(root: Path = ROOT) -> dict:
    global ROOT
    # root can be injected for tests; helpers use the same validated root.
    previous = ROOT
    ROOT = root.resolve()
    try:
        papers = []
        seen = set()
        for metadata_path in sorted((ROOT / "papers").rglob("metadata.json")):
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            uid = str(data.get("id") or "").strip().casefold()
            if not uid.startswith(("arxiv:", "doi:")) or uid in seen:
                raise ValueError(f"missing or duplicate paper ID: {metadata_path}")
            seen.add(uid)
            base = metadata_path.parent
            file_name = data.get("paper_file")
            pdf_path = _safe_relative(file_name, base) if file_name is not None else None
            existing = ROOT / pdf_path if pdf_path else None
            if existing is not None and existing.is_file() and existing.read_bytes()[:5] != b"%PDF-":
                raise ValueError(f"corrupt PDF: {pdf_path}")
            if file_name is not None and (existing is None or not existing.is_file()):
                pdf_state = "missing"
            else:
                pdf_state = "archived" if existing is not None else "link-only"
            notes = data.get("notes") or ["notes/explainer.zh-CN.md", "notes/dialogue.md"]
            if not isinstance(notes, list):
                raise ValueError(f"notes must be a list: {metadata_path}")
            papers.append({
                "paper_uid": uid,
                "title": str(data.get("title") or ""),
                "title_zh": str(data.get("title_zh") or ""),
                "authors": data.get("authors") or [],
                "year": data.get("year"),
                "category": data.get("primary_category") or "/".join(
                    part for part in (data.get("primary_domain"), data.get("topic")) if part
                ),
                "tags": data.get("tags") or [],
                "first_read": data.get("date_first_read") or data.get("first_read") or None,
                "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
                "pdf_path": pdf_path if existing is not None and existing.is_file() else None,
                "pdf_state": pdf_state,
                "pdf_sha256": _hash(existing) if pdf_state == "archived" else None,
                "source_pdf": data.get("source_pdf") or data.get("pdf_source_url"),
                "source_url": data.get("source_url") or data.get("url"),
                "arxiv_version": data.get("arxiv_version"),
                "analysis_scope": data.get("analysis_scope") or data.get("notes_status"),
                "notes": [_safe_relative(note, base) for note in notes],
            })
        return {"schema_version": 1, "papers": papers}
    finally:
        ROOT = previous


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_catalog(), ensure_ascii=False, indent=2) + "\n"
    dest = ROOT / "catalog.json"
    if args.write:
        if not dest.is_file() or dest.read_text(encoding="utf-8") != rendered:
            dest.write_text(rendered, encoding="utf-8")
        print(f"Indexed {len(json.loads(rendered)['papers'])} papers")
        return 0
    if not dest.is_file() or dest.read_text(encoding="utf-8") != rendered:
        print("catalog.json is stale; run python scripts/build_catalog.py --write")
        return 1
    print("catalog.json is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
