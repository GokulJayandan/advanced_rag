"""
Prompt templates for the RAG system.

All system prompts and task-specific templates are centralized here.
Enforces grounded generation, citation format, and safe behavior.
"""

from __future__ import annotations

# --- Main RAG Generation ---

RAG_SYSTEM_PROMPT = """You are a precise, factual assistant. You answer questions ONLY using the provided context documents.

STRICT RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. For every factual claim, include an inline citation in this exact format: [Source: filename, p.X]
   - If page number is unavailable, use: [Source: filename, section]
   - If section is also unavailable, use: [Source: filename]
3. NEVER fabricate sources, page numbers, or citations.
4. If the context does not contain enough information to answer, explicitly say: "The available documents do not contain sufficient information to answer this question."
5. If the context partially answers the question, answer what you can and clearly state what information is missing.
6. Do NOT speculate, guess, or extrapolate beyond what the evidence supports.
7. Be concise and direct."""

RAG_USER_PROMPT_TEMPLATE = """Context Documents:
{context}

---

Question: {query}

Provide a well-structured answer with inline citations for every factual claim. Use the format [Source: filename, p.X] for citations."""

# --- Query Classification ---

QUERY_CLASSIFICATION_PROMPT = """Classify the following query into exactly ONE of these categories:

- factual: Simple fact-based question (e.g., "What is X?")
- conceptual: Asks for explanation of a concept
- comparison: Compares two or more things
- multi_hop: Requires connecting information from multiple sources
- multi_part: Contains multiple distinct sub-questions
- ambiguous: Unclear or could mean multiple things
- summarization: Asks to summarize content
- procedural: Asks how to do something step-by-step

Query: {query}

Respond with ONLY the category name, nothing else."""

# --- Query Rewriting ---

QUERY_REWRITING_PROMPT = """The following query may be ambiguous or poorly worded for document retrieval. Generate 3 improved search queries that better capture the user's intent.

Original query: {query}

Context from conversation (if any): {context}

Return ONLY the 3 rewritten queries, one per line, with no numbering or extra text."""

# --- Query Decomposition ---

QUERY_DECOMPOSITION_PROMPT = """The following complex query needs to be broken into simpler sub-questions that can be answered independently.

Query: {query}

Decompose this into 2-4 focused sub-questions. Each should be a standalone question that helps answer the original query.

Return ONLY the sub-questions, one per line, with no numbering or extra text."""

# --- Evidence Sufficiency ---

EVIDENCE_SUFFICIENCY_PROMPT = """Given the following query and retrieved evidence, determine if the evidence is SUFFICIENT to answer the query fully.

Query: {query}

Evidence:
{evidence}

Respond with ONLY one of:
- SUFFICIENT: if the evidence contains enough information to answer the query
- INSUFFICIENT: if critical information is missing
- PARTIAL: if only part of the query can be answered

Follow your answer with a brief one-sentence explanation."""

# --- Answer Verification ---

ANSWER_VERIFICATION_PROMPT = """Verify whether the following answer is fully supported by the provided evidence.

Question: {query}

Answer: {answer}

Evidence:
{evidence}

Check for:
1. Claims not supported by evidence (list them)
2. Contradictions with evidence (list them)
3. Fabricated citations or sources (list them)

Respond in this exact format:
GROUNDED: yes/no
COVERAGE: high/medium/low
UNSUPPORTED_CLAIMS: [list or "none"]
CONTRADICTIONS: [list or "none"]
INVALID_CITATIONS: [list or "none"]"""

# --- Context Formatting ---

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source_info = chunk.get("filename", "unknown")
        page = chunk.get("page_number")
        section = chunk.get("section")

        source_label = f"[Document: {source_info}"
        if page and page > 0:
            source_label += f", p.{page}"
        if section:
            source_label += f", {section}"
        source_label += "]"

        context_parts.append(f"{source_label}\n{chunk.get('text', '')}")

    return "\n\n---\n\n".join(context_parts)
