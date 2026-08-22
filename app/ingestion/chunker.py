"""
Semantic-aware document chunker.

Splits parsed documents into chunks that respect semantic boundaries
(headings, paragraphs, sections) rather than arbitrary character splits.
Chunk sizes and overlap are configurable via settings.
"""

from __future__ import annotations

import hashlib

import tiktoken

from app.models.documents import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    ParsedDocument,
    ParsedElement,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Elements that indicate section boundaries
BOUNDARY_TYPES = {"Title", "Header"}


class SemanticChunker:
    """
    Chunks documents respecting semantic boundaries.

    Uses token-based sizing (via tiktoken) and configurable overlap.
    Preserves headings, sections, page information, and document hierarchy.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 60,
        tokenizer_name: str = "cl100k_base",
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._tokenizer = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _generate_chunk_id(self, document_id: str, text: str, index: int) -> str:
        """Generate a deterministic chunk ID from content."""
        content = f"{document_id}:{index}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """
        Split a parsed document into semantic chunks.

        Args:
            document: The parsed document to chunk.

        Returns:
            List of DocumentChunks with full metadata.
        """
        if not document.elements:
            logger.warning("empty_document", document_id=document.document_id)
            return []

        # Group elements into semantic sections
        sections = self._group_into_sections(document.elements)

        # Build chunks from sections
        raw_chunks = self._build_chunks(sections)

        # Create DocumentChunk objects with metadata
        chunks: list[DocumentChunk] = []
        for idx, (text, page_number, section) in enumerate(raw_chunks):
            token_count = self.count_tokens(text)
            chunk_id = self._generate_chunk_id(document.document_id, text, idx)

            chunk = DocumentChunk(
                text=text,
                metadata=ChunkMetadata(
                    document_id=document.document_id,
                    chunk_id=chunk_id,
                    filename=document.filename,
                    source_path=document.source_path,
                    page_number=page_number,
                    section=section,
                    document_type=document.document_type,
                    chunk_index=idx,
                    total_chunks=0,  # Updated after all chunks are built
                    token_count=token_count,
                ),
            )
            chunks.append(chunk)

        # Update total_chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.info(
            "chunking_complete",
            document_id=document.document_id,
            total_chunks=len(chunks),
            avg_tokens=sum(c.metadata.token_count for c in chunks) / max(len(chunks), 1),
        )

        return chunks

    def _group_into_sections(
        self, elements: list[ParsedElement]
    ) -> list[tuple[str, int | None, str | None]]:
        """
        Group elements into semantic sections.

        Returns list of (text, page_number, section_heading) tuples.
        """
        sections: list[tuple[str, int | None, str | None]] = []
        current_texts: list[str] = []
        current_page: int | None = None
        current_section: str | None = None

        for elem in elements:
            # If we hit a boundary, flush current accumulation
            if elem.element_type in BOUNDARY_TYPES and current_texts:
                combined = "\n".join(current_texts)
                if combined.strip():
                    sections.append((combined, current_page, current_section))
                current_texts = []

            if elem.element_type in BOUNDARY_TYPES:
                current_section = elem.text

            if elem.page_number is not None:
                current_page = elem.page_number

            current_texts.append(elem.text)

        # Flush remaining
        if current_texts:
            combined = "\n".join(current_texts)
            if combined.strip():
                sections.append((combined, current_page, current_section))

        return sections

    def _build_chunks(
        self, sections: list[tuple[str, int | None, str | None]]
    ) -> list[tuple[str, int | None, str | None]]:
        """
        Build token-sized chunks from semantic sections with overlap.

        If a section is larger than chunk_size, it's split at sentence
        boundaries. If smaller, adjacent sections are merged.
        """
        chunks: list[tuple[str, int | None, str | None]] = []
        buffer_texts: list[str] = []
        buffer_tokens = 0
        buffer_page: int | None = None
        buffer_section: str | None = None

        for text, page, section in sections:
            text_tokens = self.count_tokens(text)

            # If this single section exceeds chunk size, split it
            if text_tokens > self.chunk_size_tokens:
                # First flush the buffer
                if buffer_texts:
                    chunks.append(
                        ("\n".join(buffer_texts), buffer_page, buffer_section)
                    )
                    buffer_texts = []
                    buffer_tokens = 0

                # Split the large section
                sub_chunks = self._split_large_text(text, page, section)
                chunks.extend(sub_chunks)
                continue

            # If adding this section exceeds chunk size, flush buffer
            if buffer_tokens + text_tokens > self.chunk_size_tokens and buffer_texts:
                chunks.append(
                    ("\n".join(buffer_texts), buffer_page, buffer_section)
                )

                # Apply overlap: keep the last portion of the buffer
                overlap_text = self._get_overlap_text(buffer_texts)
                buffer_texts = [overlap_text] if overlap_text else []
                buffer_tokens = self.count_tokens(overlap_text) if overlap_text else 0

            buffer_texts.append(text)
            buffer_tokens += text_tokens
            if page is not None:
                buffer_page = page
            if section is not None:
                buffer_section = section

        # Flush final buffer
        if buffer_texts:
            chunks.append(
                ("\n".join(buffer_texts), buffer_page, buffer_section)
            )

        return chunks

    def _split_large_text(
        self, text: str, page: int | None, section: str | None
    ) -> list[tuple[str, int | None, str | None]]:
        """Split a large text block at sentence boundaries."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: list[tuple[str, int | None, str | None]] = []

        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size_tokens and current_sentences:
                chunks.append(
                    (" ".join(current_sentences), page, section)
                )

                # Overlap
                overlap_text = self._get_overlap_text(current_sentences)
                current_sentences = [overlap_text] if overlap_text else []
                current_tokens = self.count_tokens(overlap_text) if overlap_text else 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append((" ".join(current_sentences), page, section))

        return chunks

    def _get_overlap_text(self, texts: list[str]) -> str:
        """Get the trailing overlap portion from accumulated texts."""
        if not texts or self.chunk_overlap_tokens <= 0:
            return ""

        # Take text from the end until we hit overlap token count
        combined = " ".join(texts)
        tokens = self._tokenizer.encode(combined, disallowed_special=())

        if len(tokens) <= self.chunk_overlap_tokens:
            return combined

        overlap_tokens = tokens[-self.chunk_overlap_tokens:]
        return self._tokenizer.decode(overlap_tokens)
