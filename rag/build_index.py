"""
Ingestion CLI for Phase 3/4 ("document ingestion", "create index", "save index"). Supports two
knowledge sources:

  --kb        a structured knowledge-base JSON file (a list of `{"doc_id", "topic", "text"}`
              objects -- see `rag/data/aero_rentals_kb.json`).
  --text-file a plain free-form text file (e.g. `text.txt`), automatically split into
              retrieval-sized chunks by `chunk_text()` -- this is the ingestion path for
              "Production RAG Streaming Mode" (see docs/PRODUCTION_RAG.md), where a knowledge base
              doesn't need to be hand-authored as structured JSON first.

Either way, the result is a FAISS index + metadata sidecar written to disk via
`rag.retriever.Retriever`.

Usage:
    python -m rag.build_index \
        --kb rag/data/aero_rentals_kb.json \
        --out rag_indexes/aero_rentals \
        --embedding-model bge-small \
        --vector-db faiss

    python -m rag.build_index \
        --text-file rag/data/text.txt \
        --out rag_indexes/production \
        --embedding-model bge-small \
        --vector-db faiss
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from .retriever import Document, Retriever

# Splits on whitespace that follows a sentence-ending punctuation mark. Deliberately simple (no
# abbreviation/acronym handling) -- a few extra splits on "Dr." or "U.S." are harmless (the
# surrounding packing step in `_pack_units` just re-joins them with the next "sentence"), whereas
# missing a real sentence boundary would let one oversized chunk leak into a neighbor. Used only
# to find *candidate* split points for paragraphs that exceed `chunk_size_chars` -- normal,
# undersized paragraphs are never touched by this.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_documents(kb_path: str) -> list[Document]:
    with open(kb_path, encoding="utf-8") as f:
        raw = json.load(f)
    documents = []
    for entry in raw:
        metadata = {k: v for k, v in entry.items() if k not in ("doc_id", "text")}
        documents.append(Document(text=entry["text"], doc_id=entry["doc_id"], metadata=metadata))
    return documents


def build_index(kb_path: str, out_path: str, embedding_model: str = "bge-small", vector_db: str = "faiss") -> dict:
    """Builds and saves an index from `kb_path`. Returns a small report dict (also what the CLI
    prints), useful for notebook cells that want to assert on it (e.g. "index has N documents")."""
    documents = load_documents(kb_path)

    t0 = time.monotonic()
    retriever = Retriever(embedding_model=embedding_model, vector_db=vector_db)
    n_indexed = retriever.build_index_from_documents(documents)
    build_time_s = time.monotonic() - t0

    retriever.save_index(out_path)

    return {
        "kb_path": kb_path,
        "out_path": out_path,
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "documents_indexed": n_indexed,
        "build_time_s": build_time_s,
    }


def _char_window_split(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    """Last-resort splitter: a raw sliding window over characters, with no awareness of word or
    sentence boundaries. Only ever reached for a single "unit" (sentence, or failing that, word)
    that has no smaller natural boundary at all -- e.g. a long URL or base64 blob -- so this never
    fires on ordinary prose once `_split_oversized_paragraph` below has tried sentence and word
    boundaries first.

    `step` (not `chunk_size_chars - overlap_chars` inline) guards against a misconfigured
    `overlap_chars >= chunk_size_chars`, which would otherwise make `start` go non-increasing (or
    negative) and loop forever.
    """
    step = max(chunk_size_chars - overlap_chars, 1)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return chunks


def _pack_units(
    units: list[str], chunk_size_chars: int, overlap_chars: int, joiner: str, oversized_splitter
) -> list[str]:
    """Greedily packs `units` (sentences, or words) into chunks <= `chunk_size_chars`, joined by
    `joiner`. When a chunk would overflow, it's flushed and the *trailing* units worth up to
    `overlap_chars` are carried into the start of the next chunk, so a fact split exactly across a
    cut still has a good chance of being fully visible in at least one chunk.

    Any single `unit` that alone exceeds `chunk_size_chars` is handed to `oversized_splitter`
    (a smaller-granularity fallback -- sentences fall back to words, words fall back to raw
    characters) rather than being force-fit, so no chunk this function returns is ever longer than
    `chunk_size_chars` -- except whatever an exhausted `oversized_splitter` chain itself cannot
    shrink further (a single character-run with no whitespace at all).
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > chunk_size_chars:
            if current:
                chunks.append(joiner.join(current))
                current, current_len = [], 0
            chunks.extend(oversized_splitter(unit, chunk_size_chars, overlap_chars))
            continue

        added_len = len(unit) + (len(joiner) if current else 0)
        if current and current_len + added_len > chunk_size_chars:
            chunks.append(joiner.join(current))
            # Carry trailing units worth up to `overlap_chars` into the next chunk for continuity.
            carried: list[str] = []
            carried_len = 0
            for u in reversed(current):
                extra = len(u) + (len(joiner) if carried else 0)
                if carried_len + extra > overlap_chars:
                    break
                carried.insert(0, u)
                carried_len += extra
            current, current_len = carried, carried_len
            added_len = len(unit) + (len(joiner) if current else 0)

        current.append(unit)
        current_len += added_len
    if current:
        chunks.append(joiner.join(current))
    return chunks


def _split_by_words(paragraph: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    words = [w for w in paragraph.split(" ") if w]
    if len(words) <= 1:
        return _char_window_split(paragraph, chunk_size_chars, overlap_chars)
    return _pack_units(words, chunk_size_chars, overlap_chars, joiner=" ", oversized_splitter=_char_window_split)


def _split_oversized_paragraph(paragraph: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    """Sub-splits a single paragraph that exceeds `chunk_size_chars`, preferring sentence
    boundaries (`_SENTENCE_SPLIT_RE`) so no chunk ever cuts mid-sentence -- let alone mid-word, as
    a naive character-offset sliding window would (see docs/PRODUCTION_RAG.md's root-cause writeup
    for the real-world version of this: it fragmented a 12K-character prose document into
    incoherent half-sentence/half-URL chunks whose embeddings carried almost no retrievable
    signal). Falls back to word boundaries (`_split_by_words`), and only then to a raw character
    window (`_char_window_split`), for paragraphs with no sentence punctuation or no whitespace at
    all (e.g. one giant token).
    """
    sentences = [s for s in (p.strip() for p in _SENTENCE_SPLIT_RE.split(paragraph)) if s]
    if len(sentences) <= 1:
        return _split_by_words(paragraph, chunk_size_chars, overlap_chars)
    return _pack_units(sentences, chunk_size_chars, overlap_chars, joiner=" ", oversized_splitter=_split_by_words)


def chunk_text(
    text: str, chunk_size_chars: int = 1000, overlap_chars: int = 150, min_chunk_chars: int = 200
) -> list[str]:
    """Splits free-form text into retrieval-sized chunks, treating each blank-line-separated
    paragraph as one semantic unit:

      - A paragraph that fits within `chunk_size_chars` AND is at least `min_chunk_chars` long
        becomes its own standalone chunk -- it is never merged with a neighboring paragraph, even
        if there's room. Merging two complete, topically-distinct paragraphs together dilutes
        their combined embedding: a query about either topic ranks the merged chunk lower than it
        would rank either paragraph alone, and at a fixed `top_k` this can drop the answer out of
        the retrieved set entirely. See docs/PRODUCTION_RAG.md for a real, measured example of this
        ("What is the Yield Bull?" failing to retrieve the one chunk that answers it because it had
        been merged with the unrelated "Solana Bull" paragraph).

      - A paragraph *shorter* than `min_chunk_chars` (a header, a short "Field: value" line, a
        "----" divider, etc.) is too small to carry useful embedding signal on its own, so it IS
        packed together with adjacent equally-small paragraphs, up to `chunk_size_chars` -- this is
        the one case merging remains correct and necessary (see docs/PRODUCTION_RAG.md Section 9
        for the original bug this fixed: documents that use blank lines purely for visual
        structure used to fragment into one chunk per tiny block).

      - A paragraph *longer* than `chunk_size_chars` is sub-split via `_split_oversized_paragraph`
        (sentence-boundary-aware, with word- and character-level fallbacks) rather than a blind
        character-offset window, so long-form prose is never cut mid-sentence or mid-word.

    Blank/whitespace-only paragraphs are dropped. Returns an empty list for empty/whitespace-only
    input.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_oversized_paragraph(paragraph, chunk_size_chars, overlap_chars))
            continue

        if len(paragraph) < min_chunk_chars:
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= chunk_size_chars:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = paragraph
            continue

        # A substantial, self-contained paragraph: flush whatever small-fragment buffer preceded
        # it, then let it stand alone as its own chunk (see the docstring above for why merging it
        # with a neighbor would hurt retrieval precision).
        if buffer:
            chunks.append(buffer)
            buffer = ""
        chunks.append(paragraph)

    if buffer:
        chunks.append(buffer)
    return chunks


def load_documents_from_text_file(
    text_path: str, chunk_size_chars: int = 1000, overlap_chars: int = 150, min_chunk_chars: int = 200
) -> list[Document]:
    """Reads a plain text file and chunks it (via `chunk_text`) into `Document`s suitable for
    `Retriever.build_index_from_documents` -- the ingestion path for a free-form `text.txt`
    knowledge base, as opposed to `load_documents`'s structured KB JSON. `doc_id`s are
    `<basename>-chunk-<i>`, stable across rebuilds as long as the file's paragraph structure
    doesn't change.
    """
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(
        text, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars, min_chunk_chars=min_chunk_chars
    )
    basename = os.path.basename(text_path)
    return [Document(text=chunk, doc_id=f"{basename}-chunk-{i}") for i, chunk in enumerate(chunks)]


def build_index_from_text_file(
    text_path: str,
    out_path: str,
    embedding_model: str = "bge-small",
    vector_db: str = "faiss",
    chunk_size_chars: int = 1000,
    overlap_chars: int = 150,
    min_chunk_chars: int = 200,
) -> dict:
    """Same as `build_index`, but ingests a plain text file (chunked via `chunk_text`) instead of
    a structured KB JSON -- the entry point for "Production RAG Streaming Mode", see
    docs/PRODUCTION_RAG.md. Returns the same report shape as `build_index`, plus the chunking
    parameters used."""
    documents = load_documents_from_text_file(text_path, chunk_size_chars, overlap_chars, min_chunk_chars)
    if not documents:
        raise ValueError(f"{text_path!r} produced no chunks -- is the file empty?")

    t0 = time.monotonic()
    retriever = Retriever(embedding_model=embedding_model, vector_db=vector_db)
    n_indexed = retriever.build_index_from_documents(documents)
    build_time_s = time.monotonic() - t0

    retriever.save_index(out_path)

    return {
        "text_path": text_path,
        "out_path": out_path,
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "chunk_size_chars": chunk_size_chars,
        "overlap_chars": overlap_chars,
        "min_chunk_chars": min_chunk_chars,
        "documents_indexed": n_indexed,
        "build_time_s": build_time_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kb", help="Path to a structured knowledge-base JSON file.")
    source.add_argument("--text-file", help="Path to a plain text file, chunked automatically.")
    parser.add_argument("--out", required=True, help="Output path prefix for the saved index.")
    parser.add_argument("--embedding-model", default="bge-small")
    parser.add_argument("--vector-db", default="faiss")
    parser.add_argument("--chunk-size-chars", type=int, default=1000, help="Only used with --text-file.")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Only used with --text-file.")
    parser.add_argument(
        "--min-chunk-chars", type=int, default=200,
        help="Only used with --text-file. Paragraphs shorter than this are merged with their "
             "neighbors instead of becoming their own (context-less) chunk."
    )
    args = parser.parse_args()

    if args.kb:
        report = build_index(args.kb, args.out, args.embedding_model, args.vector_db)
    else:
        report = build_index_from_text_file(
            args.text_file, args.out, args.embedding_model, args.vector_db,
            args.chunk_size_chars, args.overlap_chars, args.min_chunk_chars,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
