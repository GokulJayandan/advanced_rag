"""Unit tests for document parsing and validation."""

import pytest
from pathlib import Path
from app.ingestion.parser import DocumentParser, sanitize_filename, generate_document_id
from app.models.documents import DocumentType


def test_sanitize_filename():
    assert sanitize_filename("safe_file.pdf") == "safe_file.pdf"
    assert sanitize_filename("../../malicious/path.pdf") == "path.pdf"
    assert sanitize_filename("file with spaces & symbols!.docx") == "file with spaces _ symbols_.docx"


def test_generate_document_id():
    id1 = generate_document_id("/path/to/doc.pdf", "hash123")
    id2 = generate_document_id("/path/to/doc.pdf", "hash123")
    id3 = generate_document_id("/path/to/doc.pdf", "hash456")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


def test_parser_validation(temp_dir):
    parser = DocumentParser(max_file_size_bytes=1024 * 1024)

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        parser.validate_file(temp_dir / "non_existent.pdf")

    # Unsupported extension
    unsupported = temp_dir / "data.exe"
    unsupported.write_text("binary")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.validate_file(unsupported)

    # Empty file
    empty = temp_dir / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="Empty file"):
        parser.validate_file(empty)

    # Valid file
    valid = temp_dir / "valid.txt"
    valid.write_text("Hello world")
    parser.validate_file(valid)
