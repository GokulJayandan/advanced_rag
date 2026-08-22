"""
Citation validation — verifies citations against actually retrieved chunks.

Checks that: the citation exists, the referenced document exists, the chunk
was actually retrieved, and page/section metadata is valid.
"""

from __future__ import annotations

from app.models.generation import Citation, CitationValidationResult
from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class CitationValidator:
    """
    Validates citations against retrieved evidence.

    Ensures no fabricated citations, source references, or page numbers
    make it into the final answer.
    """

    def validate(
        self,
        citations: list[Citation],
        retrieved_chunks: list[ScoredChunk],
    ) -> CitationValidationResult:
        """
        Validate all citations against retrieved chunks.

        Args:
            citations: Citations extracted from the generated answer.
            retrieved_chunks: The chunks that were actually retrieved and used.

        Returns:
            CitationValidationResult with per-citation validity.
        """
        # Build lookup from retrieved chunks
        available_files = set()
        available_pages: dict[str, set[int]] = {}
        available_sections: dict[str, set[str]] = {}
        chunk_id_by_file: dict[str, list[str]] = {}

        for chunk in retrieved_chunks:
            fname = chunk.metadata.filename
            available_files.add(fname)

            if fname not in available_pages:
                available_pages[fname] = set()
            if chunk.metadata.page_number is not None:
                available_pages[fname].add(chunk.metadata.page_number)

            if fname not in available_sections:
                available_sections[fname] = set()
            if chunk.metadata.section:
                available_sections[fname].add(chunk.metadata.section)

            if fname not in chunk_id_by_file:
                chunk_id_by_file[fname] = []
            chunk_id_by_file[fname].append(chunk.chunk_id)

        validated_citations = []
        valid_count = 0

        for citation in citations:
            validated = citation.model_copy()

            # Check 1: Does the referenced file exist in retrieved chunks?
            if citation.filename not in available_files:
                # Fuzzy match: check if the filename is a substring
                matched = False
                for fname in available_files:
                    if citation.filename.lower() in fname.lower() or fname.lower() in citation.filename.lower():
                        validated.filename = fname
                        matched = True
                        break

                if not matched:
                    validated.is_valid = False
                    validated.validation_error = f"Document '{citation.filename}' was not in retrieved sources"
                    validated_citations.append(validated)
                    continue

            fname = validated.filename

            # Check 2: If page is cited, is it valid?
            if citation.page_number is not None:
                if available_pages.get(fname) and citation.page_number not in available_pages[fname]:
                    validated.is_valid = False
                    validated.validation_error = (
                        f"Page {citation.page_number} not found in retrieved chunks from '{fname}'"
                    )
                    validated_citations.append(validated)
                    continue

            # Check 3: If section is cited, is it roughly valid?
            if citation.section and available_sections.get(fname):
                section_match = any(
                    citation.section.lower() in s.lower() or s.lower() in citation.section.lower()
                    for s in available_sections[fname]
                )
                if not section_match and available_sections[fname]:
                    # Soft warning, not necessarily invalid
                    validated.validation_error = f"Section '{citation.section}' not exactly matched"

            # Map to chunk ID
            if fname in chunk_id_by_file:
                validated.chunk_id = chunk_id_by_file[fname][0]

            validated.is_valid = True
            valid_count += 1
            validated_citations.append(validated)

        result = CitationValidationResult(
            total_citations=len(citations),
            valid_citations=valid_count,
            invalid_citations=len(citations) - valid_count,
            citations=validated_citations,
            all_valid=valid_count == len(citations),
        )

        logger.info(
            "citation_validation",
            total=result.total_citations,
            valid=result.valid_citations,
            invalid=result.invalid_citations,
        )

        return result
