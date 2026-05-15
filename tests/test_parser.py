"""Tests for document parsers."""

import tempfile
from pathlib import Path

from doc2graph.parser.md_parser import MdParser
from doc2graph.parser.txt_parser import TxtParser
from doc2graph.parser.pdf_parser import PdfParser
from doc2graph.parser.docx_parser import DocxParser


class TestMdParser:
    def test_parse_simple(self):
        parser = MdParser()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Hello\n\nThis is a test document.")
            f.flush()
            result = parser.parse(f.name)
            assert "Hello" in result
            assert "test document" in result

    def test_supports_md(self):
        assert MdParser.supports(Path("test.md"))
        assert MdParser.supports(Path("test.markdown"))
        assert not MdParser.supports(Path("test.pdf"))


class TestTxtParser:
    def test_parse_simple(self):
        parser = TxtParser()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Plain text content here.")
            f.flush()
            result = parser.parse(f.name)
            assert "Plain text" in result

    def test_supports_txt(self):
        assert TxtParser.supports(Path("test.txt"))
        assert not TxtParser.supports(Path("test.pdf"))
