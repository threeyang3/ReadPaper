#!/usr/bin/env python3
"""Archive redistributable PDFs without overwriting files or interrupting other papers.

An unlicensed/link-only record (paper_file=null) is a normal archive entry.
Only whitelisted HTTPS sources are followed; existing PDFs are never replaced.
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
ALLOWED_LICENSES = {"CC BY 4.0", "CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC0 1.0"}


def _approved_url(url: str) -> bool:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_HOSTS
        and port in (None, 443)
        and not parsed.username
        and not parsed.password
    )


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not _approved_url(newurl):
            raise ValueError("PDF source redirected to an unapproved URL")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str) -> bytes:
    if not isinstance(url, str) or not _approved_url(url):
        raise ValueError("unapproved PDF source URL")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ReadPaper-ResearchArchive/1.0 (personal noncommercial archive)",
            "Accept": "application/pdf",
        },
    )
    opener = urllib.request.build_opener(_SafeRedirect)
    with opener.open(req, timeout=45) as response:
        if not _approved_url(response.geturl()):
            raise ValueError("unapproved final PDF URL")
        raw = response.read(MAX_BYTES + 1)
    if not (MIN_BYTES <= len(raw) <= MAX_BYTES) or not raw.startswith(b"%PDF-"):
        raise ValueError(f"not a valid-size PDF: {len(raw)} bytes")
    return raw


def _archive(metadata_path: pathlib.Path) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pdf_name = metadata.get("paper_file")
    if pdf_name is None:
        print(f"Link-only: {metadata_path.relative_to(ROOT)} (no redistributable PDF)")
        return "skipped"
    if (
        not isinstance(pdf_name, str)
        or not pdf_name
        or pathlib.PurePath(pdf_name).name != pdf_name
        or pathlib.PurePosixPath(pdf_name).name != pdf_name
        or "\\" in pdf_name
        or not pdf_name.lower().endswith(".pdf")
        or pdf_name.startswith(".")
    ):
        raise ValueError("unsafe or invalid paper_file")
    license_id = str(metadata.get("license") or "").strip().upper()
    if license_id not in ALLOWED_LICENSES:
        raise ValueError(f"License not approved for public redistribution: {license_id}")
    dest = metadata_path.parent / pdf_name
    provenance_path = dest.with_name(dest.name + ".provenance.json")
    if dest.exists():
        with dest.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError(f"corrupted existing PDF: {dest}")
        actual_hash = _sha256(dest)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
        if isinstance(provenance, dict):
            if provenance.get("sha256") != actual_hash:
                raise ValueError("existing PDF does not match its provenance hash")
            if provenance.get("paper_uid") != metadata.get("id"):
                raise ValueError("existing PDF provenance belongs to another paper")
            if str(provenance.get("arxiv_version") or "") != str(metadata.get("arxiv_version") or ""):
                raise ValueError("PDF version changed; keep the old PDF and explicitly archive the new version")
        print(f"Already archived: {dest.relative_to(ROOT)} sha256={actual_hash}" + ("" if provenance else " (legacy: version unverified)"))
        return "skipped"

    urls = [metadata.get("source_pdf") or metadata.get("pdf_source_url")]
    urls.extend(metadata.get("alternate_pdf_urls") or [])
    for url in urls:
        if not url:
            continue
        try:
            raw = download(url)
            digest = hashlib.sha256(raw).hexdigest()
            provenance = {
                "schema_version": 1,
                "paper_uid": metadata.get("id"),
                "source_url": url,
                "arxiv_version": str(metadata.get("arxiv_version") or ""),
                "sha256": digest,
                "size_bytes": len(raw),
            }
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = pathlib.Path(tmp.name)
            try:
                # Never rewrite an existing PDF, even if a concurrent job won the race.
                if dest.exists():
                    raise FileExistsError(f"PDF appeared during download: {dest}")
                tmp_path.replace(dest)
                provenance_path.write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            print(f"Archived {dest.relative_to(ROOT)}, sha256={digest}, from {url}")
            return "archived"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            print(f"Attempt failed: {url} -> {exc}", file=sys.stderr)
    raise RuntimeError("No approved PDF URL succeeded")


def main() -> int:
    archived = skipped = 0
    failed: list[str] = []
    for metadata_path in sorted((ROOT / "papers").rglob("metadata.json")):
        try:
            status = _archive(metadata_path)
            archived += status == "archived"
            skipped += status == "skipped"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failed.append(str(metadata_path.relative_to(ROOT)))
            print(f"Archive failed for {metadata_path.relative_to(ROOT)}: {exc}", file=sys.stderr)
    print(f"PDF archive: {archived} new, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("Failed records: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
