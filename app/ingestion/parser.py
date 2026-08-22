"""
Document parser using Unstructured.

Supports PDF, DOCX, TXT, and Markdown. Extracts elements with metadata
including page numbers, sections, and element types. Validates files
for size, extension, and content safety.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models.documents import (
    DocumentType,
    ParsedDocument,
    ParsedElement,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Allowed extensions and their document types
EXTENSION_MAP: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
}


def sanitize_filename(filename: str) -> str:
    """Remove potentially dangerous characters and path traversals from filenames."""
    # Extract only basename if a path was passed
    name = Path(filename).name
    # Replace dangerous characters
    sanitized = re.sub(r'[^\w\s\-\.]', '_', name)
    sanitized = re.sub(r'\.+', '.', sanitized)
    sanitized = re.sub(r'^[_\.\-]+', '', sanitized)
    return sanitized.strip()


def generate_document_id(filepath: str, content_hash: str | None = None) -> str:
    """Generate a deterministic document ID from file path and optional content hash."""
    seed = content_hash if content_hash else filepath
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


class DocumentParser:
    """
    Parses documents into structured elements with metadata.

    Supports PDF, DOCX, TXT, and Markdown via the Unstructured library.
    Validates file size, extension, and content before processing.
    """

    def __init__(self, max_file_size_bytes: int = 50 * 1024 * 1024):
        self.max_file_size_bytes = max_file_size_bytes
        self._supported_extensions = set(EXTENSION_MAP.keys())

    def validate_file(self, filepath: Path) -> None:
        """Validate file before parsing. Raises ValueError on failure."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if not filepath.is_file():
            raise ValueError(f"Not a file: {filepath}")

        suffix = filepath.suffix.lower()
        if suffix not in self._supported_extensions:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(self._supported_extensions))}"
            )

        file_size = filepath.stat().st_size
        if file_size == 0:
            raise ValueError(f"Empty file: {filepath}")

        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File too large: {file_size / (1024*1024):.1f}MB "
                f"(max: {self.max_file_size_bytes / (1024*1024):.0f}MB)"
            )

    def _get_content_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def parse(self, filepath: str | Path) -> ParsedDocument:
        """
        Parse a document into structured elements.

        Args:
            filepath: Path to the document file.

        Returns:
            ParsedDocument with extracted elements and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file fails validation.
        """
        filepath = Path(filepath)
        self.validate_file(filepath)

        suffix = filepath.suffix.lower()
        doc_type = EXTENSION_MAP[suffix]
        content_hash = self._get_content_hash(filepath)
        doc_id = generate_document_id(str(filepath), content_hash)

        logger.info(
            "parsing_document",
            filename=filepath.name,
            document_type=doc_type.value,
            document_id=doc_id,
            file_size_bytes=filepath.stat().st_size,
        )

        try:
            elements = self._parse_with_unstructured(filepath, doc_type)
        except Exception as e:
            logger.error(
                "parse_failed",
                filename=filepath.name,
                error=str(e),
            )
            raise

        parsed = ParsedDocument(
            document_id=doc_id,
            filename=sanitize_filename(filepath.name),
            source_path=str(filepath),
            document_type=doc_type,
            elements=elements,
        )

        logger.info(
            "parse_complete",
            filename=filepath.name,
            document_id=doc_id,
            total_elements=parsed.total_elements,
        )

        return parsed

    def _parse_with_unstructured(
        self, filepath: Path, doc_type: DocumentType
    ) -> list[ParsedElement]:
        """Parse using the Unstructured library."""
        from unstructured.partition.auto import partition

        raw_elements = partition(filename=str(filepath))

        elements: list[ParsedElement] = []
        current_section: str | None = None

        for elem in raw_elements:
            # Track section headings
            category = getattr(elem, "category", "NarrativeText")
            if category == "Title":
                current_section = str(elem)

            text = str(elem).strip()
            if not text:
                continue

            # Extract page number from metadata
            meta = getattr(elem, "metadata", None)
            page_number = None
            if meta:
                page_number = getattr(meta, "page_number", None)

            elements.append(
                ParsedElement(
                    text=text,
                    element_type=category,
                    page_number=page_number,
                    section=current_section,
                    metadata={
                        "category": category,
                        "element_id": getattr(elem, "id", None),
                    },
                )
            )

        return elements
