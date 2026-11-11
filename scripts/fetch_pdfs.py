#!/usr/bin/env python3
"""Fetch source PDFs from authorized open-access URLs, verifying basic file integrity.

Runs in GitHub Actions when metadata.json is added/changed. Never modifies a PDF.
Does not use any third-party Python dependencies.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_BYTES = 30 * 1024 * 1024
MIN_BYTES = 100 * 1024
ALLOWED_HOSTS = {
    "arxiv.org",
    "export.arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
}

def download(url: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"unapproved source hostname: {parsed.hostname}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ReadPaper-ResearchArchive/1.0 (personal noncommercial archive)",
                 "Accept": "application/pdf"},
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read(MAX_BYTES + 1)
    if not (MIN_BYTES <= len(raw) <= MAX_BYTES) or not raw.startswith(b"%PDF-"):
        raise ValueError(f"not a valid-size PDF: {len(raw)} bytes, header={raw[:20]!r}")
    return raw

def main() -> int:
    count = 0
    failed = []
    for metadata_path in sorted((ROOT / "papers").rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdf_name = metadata["paper_file"]
        if pathlib.Path(pdf_name).name != pdf_name or not pdf_name.lower().endswith(".pdf"):
            raise ValueError(f"unsafe PDF filename in {metadata_path}")
        dest = metadata_path.parent / pdf_name
        if dest.exists():
            if dest.read_bytes()[:5] != b"%PDF-":
                raise ValueError(f"corrupted existing PDF: {dest}")
            print(f"Already archived: {dest.relative_to(ROOT)}")
            continue
        license_id = str(metadata.get("license", "")).strip().upper()
        if license_id not in {"CC BY 4.0", "CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC0 1.0"}:
            raise ValueError(f"License not approved for public redistribution: {license_id}")
        urls = [metadata["source_pdf"], *metadata.get("alternate_pdf_urls", [])]
        for url in urls:
            try:
                raw = download(url)
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = pathlib.Path(tmp.name)
                tmp_path.replace(dest)
                print(f"Archived {dest.relative_to(ROOT)}, bytes={len(raw)}, sha256={hashlib.sha256(raw).hexdigest()}, from {url}")
                count += 1
                break
            except (OSError, ValueError, urllib.error.URLError) as exc:
                print(f"Attempt failed: {url} -> {exc}", file=sys.stderr)
        else:
            failed.append(str(metadata_path.relative_to(ROOT)))
    print(f"New PDFs archived: {count}")
    if failed:
        print("PDF download failed for: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())

