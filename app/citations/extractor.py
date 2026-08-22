"""
Citation extraction from generated text.

Parses inline citations in the format [Source: filename, p.X] and maps
them back to retrieved chunks for validation.
"""

from __future__ import annotations

import re

from app.models.generation import Citation
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Pattern matches: [Source: filename.ext, p.123] or [Source: filename.ext, section name] or [Source: filename.ext]
CITATION_PATTERN = re.compile(
    r'\[Source:\s*([^,\]]+?)(?:,\s*p\.?\s*(\d+))?(?:,\s*([^\]]+))?\]',
    re.IGNORECASE,
)


class CitationExtractor:
    """
    Extracts inline citations from generated answer text.

    Parses citations in the format:
    - [Source: filename, p.X]
    - [Source: filename, section]
    - [Source: filename]
    """

    def extract(self, text: str) -> list[Citation]:
        """
        Extract all citations from generated text.

        Returns a list of Citation objects with filename, page, and section.
        """
        citations = []
        seen = set()

        for match in CITATION_PATTERN.finditer(text):
            filename = match.group(1).strip()
            page_str = match.group(2)
            section = match.group(3)

            page_number = int(page_str) if page_str else None
            section = section.strip() if section else None

            # Deduplicate
            key = (filename, page_number, section)
            if key in seen:
                continue
            seen.add(key)

            citations.append(Citation(
                citation_text=match.group(0),
                filename=filename,
                page_number=page_number,
                section=section,
            ))

        logger.info("citations_extracted", count=len(citations))
        return citations
