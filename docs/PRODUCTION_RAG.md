# Production RAG Streaming Mode

What this is, why it's built the way it is, what was actually validated and how, and what you
need to do to point it at your own knowledge base. This is the productionization of the one
mechanism the Mode A-F research comparison (`docs/MODE_C_IMPLEMENTATION_REPORT.md`,
`docs/ARCHITECTURE_REPORT.md`) found to actually work, not a new injection mechanism.

## 1. What it is

A standing RAG setup for the live `moshi.server` (not just `moshi.offline`'s scripted runs):

1. A plain text file (`rag/data/text.txt` by default) is automatically chunked and embedded into a
   FAISS index -- no hand-authored structured KB JSON required.
2. The live server is started with `--rag-enable --rag-injection-mode persona_rag` pointed at that
   index.
3. Per connection, a `rag_query` parameter (already supported by `moshi.server` since the Mode C
   increment) triggers one retrieval + one `<system>`-wrapped injection burst, immediately after
   the persona/voice prompt and before any user audio is processed -- then the live duplex
   conversation proceeds completely normally, indistinguishable from a connection without RAG at
   all from that point on.

## 2. Why Mode C, and only Mode C

This is a deliberate constraint, not an oversight, backed by the full A-F comparison:

| Mode | Why it's excluded from production |
|---|---|
| A (baseline) | No retrieval at all -- the thing being productionized. |
| B (naive prompt template) | Confirmed negative control (Section 6): retrieves the same facts as C but doesn't engage with them at all. |
| D (turn injection) | Real-run result (Section 8): the burst itself doesn't leak, but the model abandons its in-progress sentence and re-greets instead of grounding. |
| E (dynamic/periodic injection) | Real-run result (Section 10): confirmed the `<system>`-tag hypothesis (no re-greet) but still doesn't ground -- the injected facts have no measurable effect once generation has already started. |
| F arm 2 (reset_and_replay) | Works, but costs ~1.5x arm 1's latency for no behavioral benefit over arm 1 -- there is no reason to ever choose this in production. |
| **C / F arm 1 (this mode)** | The only mechanism that reliably grounds, **and** it never resets the live RingKVCache. |

The cross-cutting finding from D and E (Section 10) is that injection *timing* relative to
generation -- before the model has sampled any part of its response, vs. after -- is what
actually determines whether injected knowledge gets used, not the `<system>`-tag format. That is
exactly what "once per connection, before generation starts" (Mode C's policy) guarantees and
what any mid-stream policy cannot.

## 3. Why "once per user turn" means "once per connection" here

PersonaPlex has no ASR anywhere in its pipeline (`docs/MODE_C_IMPLEMENTATION_REPORT.md` Section 2)
-- the only text ever available is the model's own sampled output, never a transcript of what the
user said. There is therefore no live query text to retrieve against mid-call. "Inject once per
user turn, never mid-stream" collapses to "inject once, at connection start, using the query
supplied via the `rag_query` connection parameter" -- which is exactly Mode C's existing, already
real-pod-validated design. Building genuine per-utterance retrieval would require bolting on a
separate ASR component listening to the same PCM stream, which is explicitly out of scope (no ASR
integration).

## 4. What was built

| File | Purpose |
|---|---|
| `rag/build_index.py` | Added `chunk_text()` (paragraph-aware chunking with overlap for long paragraphs), `load_documents_from_text_file()`, `build_index_from_text_file()`, and a `--text-file` CLI option alongside the existing `--kb`. No changes to `rag/retriever.py`/`rag/vector_store.py` -- they were already format-agnostic (`Document(text, doc_id, metadata)` in, FAISS index out), so plain-text ingestion only needed a new *front door*, not new retrieval machinery. |
| `rag/data/text.txt` | The default "automatically used" knowledge base -- originally the AeroRentals facts validated in Sections 3d/6, rewritten as flowing prose paragraphs instead of structured JSON; now holds the RobotBulls company knowledge base used to validate Section 10's fixes. Replace this file's contents (or point `RAG_TEXT_KB_PATH` at a different file in the notebook's RAG configuration cell) to use your own knowledge base -- no code changes needed. |
| `rag/ws_demo_client.py` | A from-scratch Python WebSocket client for `moshi.server`'s `/api/chat` endpoint (mirrors the browser web UI's protocol exactly -- query params, handshake byte, Opus-encoded binary audio frames, text-token messages). Nothing in `rag/server_integration.py` changed -- the live server's RAG injection code path (`ServerState`/`handle_chat`, wired during the Mode C increment) already did everything this needs; this client just lets a notebook (or any Python script) drive it the same way a real user's browser would, instead of only being exercisable by hand via the web UI. |
| `PersonaPlex_RunPod_RTX5090.ipynb`, Section 22 | Builds the index from `text.txt`, launches a RAG-enabled live server, runs a real-time-paced demo query over an actual websocket connection, and prints the retrieved chunks / injection mechanism / final transcript / streaming latency as explicit proof of each success criterion. |

## 5. What was actually validated, and how

Consistent with this project's running discipline: this machine has no GPU, no CUDA, and none of
PersonaPlex's gated weights, so the real model cannot run here.

**Validated for real, right now:**
- `chunk_text`/`load_documents_from_text_file`/`build_index_from_text_file`: 10 unit tests
  (`rag/tests/test_build_index.py`), including an end-to-end ingest-then-retrieve round trip
  against the real `faiss` library (embedder monkeypatched, same pattern as
  `rag/tests/test_retriever.py` -- no network/model download needed).
- `rag/ws_demo_client.build_query_params`: 5 unit tests (`rag/tests/test_ws_demo_client.py`).
  `aiohttp`/`sphn`/`moshi` are all imported lazily inside functions specifically so this module
  (and these tests) never require any of them to be installed -- same discipline as
  `rag/embeddings.py`'s lazy `sentence_transformers` import.
- All 108 tests in `rag/tests/` pass; `moshi/moshi/offline.py`/`server.py` are unaffected (no
  changes to either file -- this feature only adds new files plus a `rag/build_index.py`
  extension).

**NOT validated here -- requires the real RunPod RTX 5090 pod (your next step):**
The actual claim this feature hinges on -- "a real websocket connection, driven by a plain
`text.txt` file, grounds its answer without interrupting streaming or resetting the connection" --
can only be checked by running Section 22 against the real server and model. The websocket
protocol implementation in `rag/ws_demo_client.py` was written by careful, exact mirroring of
`moshi/moshi/server.py`'s `handle_chat`/`opus_loop` (same message-kind bytes, same Opus codec
calls, same query parameters), not by testing it against a real server -- that mirroring could
still be wrong in a way only a real connection attempt would reveal (e.g. an Opus framing detail,
a timing assumption). Treat the first real run as the actual test of this module, not just of the
underlying (already-proven) injection mechanism.

## 6. Using your own knowledge base

Replace the contents of `rag/data/text.txt` with your own plain text (any paragraph structure
works -- `chunk_text` splits on blank lines first; each substantial paragraph -- at or above
`min_chunk_chars`, default 200 -- becomes its own chunk, never diluted by merging with a neighbor;
only paragraphs *below* that length are packed together, and only a single paragraph that alone
exceeds `chunk_size_chars` (default 1000) gets sub-split, on sentence then word boundaries), then
re-run the notebook's "Build the FAISS index from the knowledge base" cell (Section 11). To use a
different file path entirely, change `RAG_TEXT_KB_PATH` in the RAG configuration cell (Section 10).
No other code changes are required -- retrieval, injection, and the live server's RAG code path
are all already knowledge-source-agnostic, and the token-budget guard (Section 10 below) scales
automatically to whatever size knowledge base you point it at.

## 7. Performance expectations

Per Mode C's own real-pod benchmark (`docs/MODE_C_IMPLEMENTATION_REPORT.md` Section 3d/6):
retrieval + injection together cost roughly the 8-9 second range quoted in the brief for a
~5-document, ~340-token injected block at `bge-small` embedding speed -- this is a one-time,
connection-start cost, not a per-turn or per-frame cost, since the mechanism never re-injects or
resets mid-call. Larger knowledge bases or a larger embedding model will retrieve more slowly;
larger `top_k` injects more tokens (~25ms/token, per the same benchmark) up to whatever the
token-budget guard (Section 10) actually allows through. Once injection completes, streaming
proceeds at the same speed as a connection with RAG disabled -- nothing in this mechanism touches
the per-frame `opus_loop` cost.

## 8. Real-pod bug: RAG never engaged for actual browser conversations (found and fixed)

The first real-pod test of this feature was through the actual browser web UI (real microphone,
real voice), not the notebook's scripted `rag.ws_demo_client` demo. The model gave generic,
ungrounded answers -- the symptom looked like `text.txt` wasn't "properly processed", but the
index and retrieval pipeline were both fine.

**Root cause**: `moshi.server`'s `rag_query` connection parameter only exists because *this
project* added it (Section 5's Mode C increment) -- the browser web UI predates it and has no
field to send one. PersonaPlex has no ASR, so the browser UI has genuinely no text to put there
even if it tried. `handle_chat`'s old code guarded injection on `rag_query` being truthy
(`if self.rag_session is not None and rag_query:`), so for every real conversation through the
browser, that condition was always false and RAG silently never engaged -- only the scripted demo
(which explicitly sets `rag_query=AERO_QUESTION_TEXT`) ever exercised it. This had been latent
since the Mode C increment; Section 22's own demo cell never caught it because it always supplies
an explicit query.

**Fix** (three changes, all in already-existing code paths, no new mechanism):

1. `RAGSession._retrieve_for_injection` (`rag/server_integration.py`) now accepts a falsy `query`
   and falls back to `Retriever.retrieve_all()` -- injecting knowledge-base chunks regardless of
   relevance -- instead of skipping injection. New `FaissVectorStore.get_all()`/
   `Retriever.retrieve_all()` (`rag/vector_store.py`, `rag/retriever.py`) support this by reading
   back stored chunks directly, bypassing similarity search entirely (there is nothing to rank
   against without a query). (This originally capped at `top_k`; Section 9 found and fixed a real
   bug in that cap.)
2. `moshi/moshi/server.py`'s `handle_chat` no longer requires `rag_query` to be truthy before
   attempting injection. A new `RAGConfig.default_query` / `--rag-default-query` lets an operator
   configure a real similarity-search fallback (e.g. a one-line description of the deployment's
   domain) for connections that don't supply their own query; when that's also empty, the
   whole-KB fallback above is what actually fires for a real browser connection.
3. `moshi/moshi/offline.py`'s `--rag-enable` no longer hard-requires `--rag-query` for
   `persona_rag`/`prompt_rag`/`cache_aware` (it still does for `turn_injection`/`dynamic_runtime`,
   whose "prepare" methods retrieve directly, without this fallback).

Section 22 of the notebook gained a verification cell ("Verify the fix: a connection with NO
query") that connects with `rag_query=""` -- exactly what the browser sends -- and asserts
injection still happened, instead of only ever testing the easy case.

**The remaining, genuine limitation**: this fix makes the model *always have* the knowledge base
in context, which is sufficient grounding for a knowledge base small enough to fit. It is **not**
true per-question retrieval -- there is still no live signal of what the user actually asked to
rank chunks against. For a knowledge base too large to inject in full, the practical levers are a
well-chosen `--rag-default-query` (static, but at least relevance-ranked) or, beyond the scope of
this project as currently constrained, adding ASR -- and even then, Sections 8/10's real-pod
findings suggest a turn-boundary-triggered mid-call injection would likely still arrive too late to
influence the response. There is currently no way around this without changing the "no ASR, no
mid-stream injection" constraints this project was built under.

## 9. Second real-pod bug: chunking fragmentation + a stale `top_k` cap silently dropped later entries

After Section 8's fix, a multi-entry knowledge base (Bangladesh political leadership: President,
Prime Minister, Speaker, Leader of the Opposition, four separate "Entity: .../Question: .../
Answer: ..." sections in `text.txt`) only ever grounded the **first** entry in the file, no matter
which entry was asked about. Reordering the file changed *which* entry worked, always the first
one -- a clean, deterministic, position-dependent symptom, not the vaguer "model sometimes
hallucinates" pattern Section 8's bug produced.

**Root cause, confirmed by direct reproduction against the real file**: two compounding bugs in
this project's own code, not a model limitation.

1. **Chunking was too fine-grained.** `chunk_text`'s original policy was "one paragraph (one
   blank-line-separated block) = one chunk." The Bangladesh file uses blank lines liberally for
   visual structure -- a title, four "Entity: ..." blocks, eight "Question:/Answer:" blocks, and
   three "----" dividers -- 14 blocks total for 4 conceptual entries. Run through the old
   `chunk_text`, the President's content alone (title + its Entity block + its two Q/A blocks +
   the first divider) occupied chunks 0-4.
2. **The no-query fallback was capped at `top_k` (default 5).** `RAGSession._retrieve_for_injection`
   called `Retriever.retrieve_all(limit=self.config.top_k)`, and `retrieve_all` returns chunks in
   plain file order. With 14 chunks and a cap of 5, only chunks 0-4 -- all about the title and the
   President -- were ever injected. The Prime Minister/Speaker/Opposition chunks (5-13) were never
   injected, period, regardless of which question was asked. Reordering the file just moved which
   entry landed in the surviving first 5 -- exactly the observed symptom.

Verified directly: running the real `rag/data/text.txt` through the old `chunk_text` produced
exactly 14 chunks, with chunks 0-4 covering only the title and President -- reproducing the bug
from the actual file content, not a synthetic approximation.

**Fix (two parts, addressing each root cause -- not a workaround for either):**

1. **`chunk_text` now merges consecutive short paragraphs** (`rag/build_index.py`) instead of
   giving every blank-line-separated block its own chunk: paragraphs are packed together up to
   `chunk_size_chars`, only starting a new chunk once the next paragraph would overflow the
   budget, and only sub-splitting a single paragraph that alone exceeds the budget. Run against
   the real `text.txt`, this collapses the same 14 blocks into 2 chunks. (A latent, unrelated bug
   in the sub-split loop -- `overlap_chars >= chunk_size_chars` made the scan position go
   non-increasing and loop until `MemoryError` -- was found and fixed in the same pass, via a
   `step = max(chunk_size_chars - overlap_chars, 1)` guard.)
2. **The no-query fallback is no longer capped by `top_k`.** New `RAGConfig.full_kb_max_chunks`
   (`int | None`, default `None`) is what `_retrieve_for_injection` passes to `retrieve_all()`
   now. `top_k` bounds a *ranked* similarity-search result, where cutting the lowest-ranked tail is
   reasonable; the no-query path has no ranking at all (chunks come back in plain file order), so
   reusing `top_k` as its cap was the wrong default -- it silently and deterministically drops
   whichever chunks happen to come later in the source document. The new default (`None`,
   uncapped) injects the entire knowledge base; set `--rag-full-kb-max-chunks` /
   `--rag-default-query` explicitly only once a knowledge base is large enough that injection
   latency (~25ms/token) becomes the actual constraint.

Either fix alone would have resolved this specific file (2 merged chunks comfortably fit under the
old `top_k=5` cap; an uncapped fallback would have injected all 14 fragments even unmerged) -- both
are implemented because each addresses a real, independent correctness gap: fragmentation produces
more chunks than necessary regardless of any cap, and capping the no-query path by a
similarity-search knob is conceptually wrong regardless of how well-chunked the source document is.

12 new/updated unit tests cover both fixes directly, including a reproduction of the
many-blank-line-block fragmentation pattern and the `overlap_chars >= chunk_size_chars` regression.
122 tests pass total.

## 10. Third real-pod bug: a ~12K-character document silently overflowed the model's own attention window

Replacing the small Bangladesh KB with a ~12,264-character RobotBulls company document (long-form
prose paragraphs, not short "Entity:/Question:/Answer:" blocks) reintroduced the same symptom
Section 8/9 had already fixed once: the assistant ignored relevant sections of the document and
fell back to its own pretrained knowledge, inconsistently across questions. Section 9's fixes were
still in place and correct for the Bangladesh KB's shape -- this was a different root cause,
specific to a document whose total size approaches the size of the model's own context window.

**Root cause, confirmed by direct reproduction against the real file -- two compounding bugs:**

1. **The "merge consecutive paragraphs" policy from Section 9, applied to dense prose, diluted
   embeddings instead of protecting them.** Section 9's fix was tuned for documents that use blank
   lines for *visual* structure (headers, one-line "Field: value" fragments) where each individual
   paragraph carries no standalone meaning. RobotBulls' `text.txt` is the opposite: almost every
   paragraph is already a complete, topically distinct ~150-300 word answer ("The Yield Bull is
   ...", "The Solana Bull is ..."). The old `chunk_text` still packed any two such paragraphs
   together whenever they fit under `chunk_size_chars` (800) -- which most adjacent pairs did.
   Reproduced directly: a query for **"What is the Yield Bull?"** failed to retrieve the Yield Bull
   paragraph at all (not even in the top 5) because it had been merged into one chunk with the
   unrelated "Solana Bull" paragraph, diluting the resulting embedding enough that several
   *unrelated* single-topic chunks ranked higher.
2. **The oversized-paragraph fallback cut text at raw character offsets, including mid-word and
   mid-URL.** Most paragraphs in this document exceed `chunk_size_chars`, so nearly every paragraph
   hit the sub-split path -- which sliced at a fixed character count regardless of word or sentence
   boundaries. Reproduced directly: the first paragraph was cut into `"...founded approximately in
   2020 and"` / `"y/robot-bulls, X (Twitter) at https://x.com/..."` -- the second fragment is mostly
   a URL salad with almost no retrievable semantic signal.
3. **No token-budget guard existed anywhere between retrieval and injection.** The live model's
   attention `RingKVCache` has a *fixed* capacity (`context=3000` frames for the released
   PersonaPlex checkpoint, see `moshi/moshi/models/loaders.py`) shared by the persona prompt, the
   voice prompt, the injected RAG knowledge, and the live conversation that follows -- it is a ring
   buffer, so once full, injecting more forced tokens silently evicts the *oldest* ones. A
   ~12,264-character document is on the order of ~3,000 tokens by itself -- comparable to the
   model's entire context capacity -- yet the no-query "inject everything" fallback (Section 8,
   the path every real browser/voice connection relies on) injected it uncapped, with no
   relationship between "how many chunks" and "how many forced-token frames that actually costs."
   `rag/logging_utils.inspect_kv_cache` already *measured* the cache's fill fraction for
   observability, but nothing used that measurement to guard against overflow before injecting.

**Fix (three parts, addressing each root cause):**

1. **`chunk_text` (`rag/build_index.py`) no longer merges two paragraphs that are each
   individually substantial.** A new `min_chunk_chars` parameter (default 200) draws the line: a
   paragraph at or above this length becomes its own standalone chunk, never merged with a
   neighbor; only paragraphs *below* it (the genuine "header"/"field fragment" case Section 9 was
   built for) still get packed together. Re-running the Yield Bull/Solana Bull example now keeps
   each as its own chunk, and "What is the Yield Bull?" retrieves the correct chunk as the #1 hit.
2. **Oversized paragraphs are now sub-split on sentence boundaries first, then word boundaries,
   and only fall back to a raw character window for content with no whitespace at all** (e.g. a
   long URL) -- see `_split_oversized_paragraph`/`_split_by_words`/`_char_window_split`. No chunk
   produced from the real `text.txt` cuts mid-word anymore.
3. **A token-budget guard now sits between retrieval and every injection mode** (`RAGSession
   ._compute_injection_token_budget`/`_select_within_budget` in `rag/server_integration.py`).
   Before any knowledge block is built (Mode B/C/F's shared `_retrieve_for_injection`, and Mode
   D/E's `prepare_*` methods), candidate chunks are measured with the connection's real tokenizer
   (`TokenInjector.count_tokens`) and greedily kept by score until the available budget is
   reached, then restored to original document order. The budget itself is computed live from
   `inspect_kv_cache`: `attention_capacity_frames - attention_frames_used -
   injection_reserve_frames` (new `RAGConfig.injection_reserve_frames`, default 400 frames ≈ 32s),
   so it automatically accounts for however much the persona/voice prompt already used --
   overridable via `RAGConfig.max_injection_tokens` for a deterministic cap independent of live
   cache state. This is the one place every injection mode funnels through, so the live attention
   window can never be overflowed regardless of knowledge base size or injection mode.

22 new unit tests cover all three fixes directly, including reproductions of the mid-word cut and
the dilutive-merge failure against realistic paragraph text, plus a fake-RingKVCache stand-in
(`FakeLMGenWithCache`) exercising the live token-budget computation end-to-end. 142 tests pass
total.

## 11. Scope enforcement: decline instead of hallucinating for out-of-scope questions

Sections 8-10 fix retrieval and injection so the right facts reliably make it into the live
model's context. None of that, by itself, stops the model from blending in its own pretrained
knowledge for whatever the knowledge base *doesn't* cover -- injecting facts only ever adds
context; it never tells the model not to fall back on what it already knows. Asking a RobotBulls
deployment "What is the capital of France?" would retrieve and inject *some* chunks (FAISS always
returns its nearest neighbors, however irrelevant) and the model, with no instruction otherwise,
would happily answer from its own knowledge instead of recognizing the question is out of scope.

**Fix: every injected knowledge block now carries an explicit scope instruction, and an explicit
question that retrieval can't answer gets an explicit decline instruction instead of silence.**

`RAGConfig.strict_scope` (default `True`) and `RAGConfig.refusal_message` (default a generic
phrase; set this to something specific to your deployment, e.g. naming your company) drive two
changes in `rag/server_integration.py`, applied uniformly across every injection mode (B/C/D/E/F):

1. **`rag.injection_manager.build_scoped_knowledge_block`** wraps every retrieved knowledge block
   with: *"You must answer ONLY using the information provided below... If the user's question is
   not covered by this information, respond only with: \"<refusal_message>\""* -- this is what
   makes the model decline rather than blend in its own knowledge for a question the knowledge
   base doesn't cover, even when (per the architecture's no-ASR constraint -- Section 3) the model
   was never told what the user's specific question was going to be.
2. **`rag.injection_manager.build_out_of_scope_notice`** is injected INSTEAD of nothing whenever
   retrieval comes back completely empty for an explicit query (a `rag_query`/`--rag-query` that
   scored below `score_threshold` against every indexed chunk) or the knowledge base itself has no
   documents. Previously, this case (`_retrieve_for_injection` returning no contexts) caused the
   caller to skip injection entirely, silently leaving the model with no grounding and no
   instruction at all for that call -- exactly free to answer from its own knowledge. Now it
   injects an explicit "this question isn't covered, decline" instruction instead.

`score_threshold` (`RAGConfig.score_threshold`, default `None`) is a SEPARATE, optional knob that
hard-gates the explicit-query path (#2 above) before injection even happens. It is deliberately
NOT given an aggressive default: measured directly against this project's RobotBulls
`text.txt`/`bge-small` combination, clearly off-topic questions ("What's the capital of France?")
scored ~0.42-0.56 cosine similarity, while genuine but generically-phrased in-scope questions
("How can I contact support?") scored as low as ~0.51 -- the two ranges OVERLAP, so any cutoff
aggressive enough to reliably block off-topic questions will also occasionally false-decline a
real, in-scope one. Mechanism #1 (the always-on instruction wrapper) is the primary, more reliable
defense for exactly this reason: it lets the model itself judge whether the retrieved facts
actually answer the question, rather than a single similarity number pre-deciding that with no
visibility into what was actually asked. Set `score_threshold` only after measuring your own
knowledge base's score distribution the same way (see the notebook's Section 12 verification
cell, which now prints both in-scope and out-of-scope sample scores for exactly this purpose).

New CLI flags (`moshi.server`/`moshi.offline`): `--rag-score-threshold`, `--rag-no-strict-scope`
(disables scope enforcement, restoring the pre-Section-11 behavior), `--rag-refusal-message`.

18 new unit tests cover both mechanisms across every injection mode, plus the config validation
for an empty `refusal_message` with `strict_scope` enabled. 160 tests pass total.

## 12. Fourth real-pod bug: in-document questions still answered incompletely after Section 11

After Section 11's fix, out-of-scope questions correctly got declined -- but questions that ARE
covered by `text.txt` were still sometimes unanswerable or answered from chunks that read like a
relevance-shuffled jumble rather than the source document's own structure. Two more bugs,
confirmed by direct measurement against the real RobotBulls knowledge base:

1. **`injection_reserve_frames`'s default (400 frames) was large enough to actively drop
   document content.** Measured directly: the full `text.txt` (21 chunks) needs ~2,400-2,550
   tokens including the Section 11 scope instruction (cross-checked with two independent
   tokenizers -- bge-small's BERT wordpiece gives 2,456 + 66 + 8 ≈ 2,530; GPT-2's BPE gives 2,371,
   same ballpark) against a 3,000-frame model context. Subtracting a plausible persona+voice
   prompt cost (~100-165 frames, computed from `LMModel.step_system_prompts`'s actual frame
   accounting: 1 frame per voice-prompt audio frame + 2×6 silence frames + 1 frame per persona
   text token) and then the 400-frame reserve left as little as ~2,435-2,500 tokens of budget --
   *less* than the ~2,530 needed for the whole document. `RAGSession._select_within_budget`
   correctly capped to that budget (doing exactly what it was built to do), but doing so dropped
   the lowest-ranked chunks: in a real run with `RAG_DEFAULT_QUERY` set to a generic company
   summary, the BTC Bull/ETH Bull/Solana Bull product chunks ranked lowest and were the ones cut
   -- so questions about exactly those products went unanswered, even though they're in the
   document. **Fix:** lowered the default to 100 frames (~8s) -- comfortably small enough that
   the whole RobotBulls document now fits (verified: 0 chunks dropped, all 21 present) while
   still leaving some headroom for the conversation that follows. See
   `RAGConfig.injection_reserve_frames`'s docstring for the full numbers. This is a tradeoff, not
   a magic fix: a knowledge base meaningfully larger than this one will still need either a
   smaller reserve, a more targeted `RAG_DEFAULT_QUERY`, or accepting that the lowest-relevance
   chunks won't fit -- there is no way around the model's fixed context size.

2. **The kept subset, even when everything fit, was reassembled in similarity-score order, not
   document order.** `Retriever.retrieve_context` returns FAISS top-k results already sorted by
   descending similarity to the query -- confirmed directly: querying the generic
   `RAG_DEFAULT_QUERY` against the real index returned the "RBT token" chunk first and the
   "Solana Bull" chunk last, completely unrelated to their order in `text.txt`.
   `_select_within_budget`'s "restore original order" step sorted the kept subset by its
   *position in that already-score-sorted list* -- which doesn't restore document order at all
   when the input wasn't in document order to begin with (it only happened to work in this
   project's own unit tests, whose hand-written fixtures passed already-document-ordered
   `contexts` with separately-varied `scores` -- not what real retrieval returns). **Fix:**
   `Retriever.retrieve_context`/`retrieve_all` now also return `"ids"` (the vector store's
   insertion-order integer ids, i.e. each chunk's real position in the source document --
   `build_index_from_documents` adds chunks in document order, so id order IS document order).
   `_select_within_budget` now sorts the kept subset by `ids` (falling back to list position only
   when a caller's retriever doesn't supply them, e.g. test stand-ins) -- verified directly: the
   same query that previously returned "RBT token, ..., Solana Bull" now reassembles into the
   document's own paragraph order ("company overview, ..., milestones") end to end.

2 new unit tests (plus updates to existing `retrieve_context`/`retrieve_all` shape assertions for
the new `"ids"` key) cover both fixes, including a regression test using already-score-sorted
input (mirroring real FAISS output) to prove document order is restored via `ids`, not list
position. 162 tests pass total.

## 13. Fifth real-pod bug: the model over-refused real, in-document questions

With Sections 8-12's fixes in place, retrieval was confirmed accurate (clean score separation
between in-scope and out-of-scope questions on the real RobotBulls index) and the entire document
was confirmed to fit in the injected context, in document order. Despite that, the live model
still sometimes responded "I don't know" / "that's outside my scope" / "I can't share that
information" to questions that ARE answered in `text.txt`.

**Root cause: the scope instruction itself (Section 11) over-corrected.** Its original wording led
with the restriction -- *"You must answer ONLY using the information provided below. Do not use
any other knowledge you may have, and do not guess or make up an answer. If the user's question is
not covered by this information, respond only with: \"<refusal_message>\""* -- before the model had
even read the facts below it. For a 7B streaming model conditioned that heavily toward caution,
declining is the simplest, lowest-risk completion whenever it isn't immediately certain the
knowledge block answers the question -- especially once that block runs to ~2,500 tokens (the
whole RobotBulls document, per Section 12) and the relevant fact isn't a close wording match for
how the question was phrased. The instruction that was meant to stop hallucination on out-of-scope
questions was simultaneously biasing the model toward *refusing in-scope ones* -- the opposite
failure mode, but the same underlying cause (an instruction that makes "decline" the path of least
resistance).

**Fix:** rewrote `rag.injection_manager.build_scoped_knowledge_block` to lead with confident,
complete usage of the knowledge instead of the restriction, and to explicitly forbid the exact
failure mode observed:

> The following is your complete knowledge base. Read all of it carefully and use it to answer the
> user's questions fully, accurately, and confidently -- most questions the user asks will be
> answered by something below, even if the wording differs from how they phrase it. Never say you
> don't know, refuse to answer, or claim information is unavailable if the answer is actually
> present below -- search the entire passage before deciding it isn't covered. Do not use any
> knowledge from outside this passage. Only if the user asks about something this passage truly
> does not address at all, respond only with: "\<refusal_message\>"

The hard constraint from Section 11 (never blend in pretrained knowledge) is unchanged -- only the
framing and emphasis moved, from "restrict, then mention the facts" to "use the facts confidently,
restrict only as a narrow fallback." `build_out_of_scope_notice` (used only when retrieval finds
literally nothing relevant -- a separate, already-confirmed-empty case) is unchanged.

This is a prompt-wording fix, not something verifiable by a unit test against the real model (no
GPU/model weights are available in this development environment) -- the existing tests assert the
new wording's key phrases are present (`rag/tests/test_injection_manager.py`). Confirming the
actual effect on live model behavior requires running the updated notebook against the real
checkpoint; if over-refusal persists, the next lever to try is narrowing `RAG_DEFAULT_QUERY` (a
more specific connection-start query measurably changes which facts rank highest within the
budget, per Section 12) before further adjusting this wording.

## 14. Sixth real-pod bug: streaming silently "freezes" after several minutes with no error

Reported symptom: after roughly 4-5 minutes of continuous live conversation, the assistant stops
responding entirely -- no further audio or text, no error visible anywhere, and the only recovery
is starting an entirely new connection. Notably, 4-5 minutes is suspiciously close to how long it
takes the model's 3000-frame attention window to fill at 12.5Hz (240s = 4 minutes) -- even sooner
once Section 12's fix is deliberately maximizing how much of the knowledge base gets front-loaded
into that same window at connection start.

**Root cause, confirmed by code inspection (`moshi/moshi/server.py`): exceptions inside the
connection's three concurrent loops were silently discarded, not a hang.** `handle_chat` runs
`recv_loop`, `opus_loop`, and `send_loop` as three `asyncio` tasks and waits on them with
`asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)`. Two problems compounded:

1. `opus_loop` (where every real per-frame `lm_gen.step()` call happens) had **no exception
   handling at all**, and `recv_loop` had a `try/finally` with no `except` -- any exception raised
   inside either (a shape assertion, a tensor/CUDA error, anything) propagated up into the
   `asyncio.Task` as an *unhandled* exception rather than being caught.
2. When `asyncio.wait(..., return_when=FIRST_COMPLETED)` returns, the code only ever inspected
   `pending` (the tasks NOT yet done, which it cancels) -- it never called `.exception()` on the
   task in `done` that actually finished (because it crashed). The exception was therefore never
   logged, never raised, never used to decide how to close the socket -- `ws.close()` ran
   unconditionally as if the connection had ended normally.

Net effect: whatever the real per-frame exception was (RingKVCache wraparound itself was
inspected directly and is NOT the bug -- its modulo-indexed read/write and position math in
`moshi/moshi/modules/transformer.py`'s `RingKVCache.complete()` is correct; depformer state,
rotary embeddings, and CUDA-graph argument handling were also inspected and ruled out), the
connection silently stopped producing output while the websocket stayed open, looking exactly like
a "freeze" with no diagnosable cause -- because the cause was being thrown away, not because
anything actually hung.

**Fix:** `recv_loop`/`opus_loop`/`send_loop` now each catch `Exception`, log the full traceback via
`clog`, and set `close = True` so the other two loops wind down too instead of one dying silently
while the others idle forever. `handle_chat` also now checks `task.exception()` for every task in
`done` (not just cancelling `pending`) and logs it. This does not change what causes the
underlying exception (still unknown -- the swallowing made it unobservable before now) -- it
converts an indefinite, unrecoverable, silent "freeze" into an immediate, logged, clean connection
close, which is both a real fix (the client can detect the close and reconnect, instead of waiting
forever on a connection that will never respond again) and what makes the actual root cause
diagnosable: the next occurrence will have a full traceback in the server log instead of nothing.
Section 12's reserve-frames change makes the attention window fill sooner, which may make whatever
the underlying exception is reproduce sooner/more often too -- if the logged traceback points back
to RAG's burst-injection path specifically, that is the next place to look.
