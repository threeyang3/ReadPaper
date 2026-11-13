"""Offline regression tests: never contact PDF servers in the test suite."""
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / "papers" / "robotics" / "example" / "metadata.json"
        self.source.parent.mkdir(parents=True)
        self.downloader = _load("fetch_pdfs")
        self.downloader.ROOT = self.root
        self.catalog = _load("build_catalog")
        self.catalog.ROOT = self.root

    def _metadata(self, **changes):
        data = {
            "id": "arxiv:2503.03726",
            "title": "Test paper",
            "paper_file": None,
            "source_pdf": None,
            "license": "unknown",
        }
        data.update(changes)
        self.source.write_text(json.dumps(data), encoding="utf-8")

    def test_link_only_is_not_a_failure_or_download(self):
        self._metadata()
        with mock.patch.object(self.downloader, "download", side_effect=AssertionError("network access")):
            self.assertEqual(self.downloader.main(), 0)
        catalog = self.catalog.build_catalog(self.root)
        self.assertEqual(catalog["papers"][0]["paper_uid"], "arxiv:2503.03726")
        self.assertEqual(catalog["papers"][0]["pdf_state"], "link-only")

    def test_existing_pdf_and_versioned_provenance(self):
        self._metadata(paper_file="paper.pdf", source_pdf="https://arxiv.org/pdf/2503.03726",
                       license="CC BY 4.0", arxiv_version="v2")
        payload = b"%PDF-" + b"p" * 100_000
        with mock.patch.object(self.downloader, "download", return_value=payload):
            self.assertEqual(self.downloader.main(), 0)
        pdf = self.source.parent / "paper.pdf"
        self.assertEqual(pdf.read_bytes(), payload)
        self.assertEqual(self.downloader.main(), 0)
        self._metadata(paper_file="paper.pdf", source_pdf="https://arxiv.org/pdf/2503.03726",
                       license="CC BY 4.0", arxiv_version="v3")
        self.assertEqual(self.downloader.main(), 1)
        self.assertEqual(pdf.read_bytes(), payload)

    def test_failure_isolation_and_traversal_rejection(self):
        self._metadata(paper_file="../outside.pdf", license="CC BY 4.0")
        other = self.root / "papers" / "robotics" / "other" / "metadata.json"
        other.parent.mkdir(parents=True)
        other.write_text(json.dumps({"id": "doi:10.1/demo", "paper_file": None}), encoding="utf-8")
        self.assertEqual(self.downloader.main(), 1)
        with self.assertRaises(ValueError):
            self.catalog.build_catalog(self.root)

    def test_redirects_to_unapproved_hosts_rejected(self):
        self.assertTrue(self.downloader._approved_url("https://arxiv.org/pdf/2503.03726"))
        self.assertFalse(self.downloader._approved_url("http://arxiv.org/pdf/2503.03726"))
        self.assertFalse(self.downloader._approved_url("https://127.0.0.1/path"))
        self.assertFalse(self.downloader._approved_url("https://arxiv.org:8080/path"))


if __name__ == "__main__":
    unittest.main()
