"""
Dynamic evaluation dataset generator.

Generates ground-truth evaluation questions, answers, and source citations
automatically from any ingested documents in ChromaDB using LLMs.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset, EvalItem
from app.generation.provider_manager import ProviderManager
from app.observability.logging import get_logger
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)

GENERATE_QA_PROMPT = """You are an automated RAG evaluation dataset generator.
Given the following document chunk, generate {num_questions} distinct evaluation question-and-answer pairs based STRICTLY on the facts presented in the text.

Document Filename: {filename}
Section / Page: {section} (Page {page})
Chunk Text:
\"\"\"
{text}
\"\"\"

Format your response as a valid JSON array of objects with the following schema:
[
  {{
    "query": "The question to ask",
    "expected_answer": "The concise factual answer found in the text",
    "tags": ["factual", "topic_tag"]
  }}
]

Return ONLY valid raw JSON without any markdown ticks or surrounding text."""


class DynamicEvalGenerator:
    """
    Dynamically generates evaluation questions and answers directly
    from uploaded documents in the vector database.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        provider_manager: ProviderManager,
    ):
        self.vector_store = vector_store
        self.provider_manager = provider_manager

    def generate_from_store(
        self,
        num_chunks_to_sample: int = 10,
        questions_per_chunk: int = 2,
        output_filepath: str | Path | None = None,
    ) -> EvaluationDataset:
        """
        Sample chunks from the database and dynamically generate QA pairs.
        """
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            logger.warning("no_chunks_in_store_for_eval_generation")
            return EvaluationDataset()

        import random
        sampled = random.sample(all_chunks, min(num_chunks_to_sample, len(all_chunks)))

        dataset = EvaluationDataset()

        for chunk_id, text, meta in sampled:
            if len(text.strip()) < 50:
                continue

            prompt = GENERATE_QA_PROMPT.format(
                filename=meta.filename,
                section=meta.section or "General",
                page=meta.page_number or "N/A",
                text=text,
                num_questions=questions_per_chunk,
            )

            res = self.provider_manager.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1000,
            )

            if res.error or not res.text.strip():
                continue

            try:
                # Clean up json if wrapped in ```json ... ```
                raw = res.text.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                
                qa_list = json.loads(raw.strip())
                for item in qa_list:
                    eval_item = EvalItem(
                        query=item["query"],
                        expected_answer=item["expected_answer"],
                        expected_sources=[meta.filename],
                        expected_chunk_ids=[chunk_id],
                        tags=item.get("tags", ["dynamic", "generated"]),
                    )
                    dataset.add_item(eval_item)

            except Exception as e:
                logger.warning("failed_parsing_generated_qa", error=str(e))

        if output_filepath and len(dataset) > 0:
            dataset.save(output_filepath)
            logger.info("dynamic_dataset_saved", path=str(output_filepath), count=len(dataset))

        return dataset
