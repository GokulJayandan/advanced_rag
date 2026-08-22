"""Unit tests for CitationValidator and CitationExtractor."""

from app.citations.extractor import CitationExtractor
from app.citations.validator import CitationValidator


def test_citation_extractor():
    extractor = CitationExtractor()
    text = (
        "Dense retrieval uses embeddings [Source: rag_overview.pdf, p.12]. "
        "BM25 relies on keywords [Source: sparse_guide.docx, Architecture Section] "
        "and overall [Source: general.md]."
    )
    citations = extractor.extract(text)
    assert len(citations) == 3
    assert citations[0].filename == "rag_overview.pdf"
    assert citations[0].page_number == 12
    assert citations[1].filename == "sparse_guide.docx"
    assert citations[1].section == "Architecture Section"
    assert citations[2].filename == "general.md"


def test_citation_validator_valid_and_invalid(sample_scored_chunks):
    extractor = CitationExtractor()
    validator = CitationValidator()

    text = (
        "Valid point [Source: sample_guide.pdf, p.1]. "
        "Fake document claim [Source: fabricated_source.pdf, p.99]."
    )
    citations = extractor.extract(text)
    result = validator.validate(citations, sample_scored_chunks)

    assert result.total_citations == 2
    assert result.valid_citations == 1
    assert result.invalid_citations == 1
    assert result.all_valid is False
    assert result.citations[0].is_valid is True
    assert result.citations[1].is_valid is False
