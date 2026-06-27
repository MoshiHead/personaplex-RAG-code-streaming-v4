"""
Central configuration for the PersonaPlex RAG research framework.

Importing this module has zero side effects and zero heavy dependencies (no torch, faiss,
sentence-transformers, etc.), so it is always safe to import even when ENABLE_RAG=False --
this is what lets the RunPod notebook expose RAG configuration variables unconditionally
without affecting baseline PersonaPlex startup.

See docs/ARCHITECTURE_REPORT.md (Section 6) and docs/STREAMING_AND_INJECTION_DESIGN.md for the
reasoning behind each mode.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import os


class InjectionMode(str, Enum):
    """Injection strategies under research. String-valued so notebook widgets / env vars / JSON
    logs can use plain strings without an extra encode/decode step."""

    BASELINE = "baseline"               # Mode A -- no RAG, pure PersonaPlex. Always supported.
    PROMPT_RAG = "prompt_rag"           # Mode B -- negative-control baseline: naive "Relevant
                                         # Knowledge: ... User Question: ..." block. Expected to
                                         # underperform Mode C; kept to *measure* that gap, not to
                                         # be tuned into working well.
    PERSONA_RAG = "persona_rag"         # Mode C -- knowledge folded into the same <system>...<system>
                                         # mechanism PersonaPlex uses for its own persona prompt.
    TURN_INJECTION = "turn_injection"   # Mode D -- inject once per detected end-of-user-turn.
    DYNAMIC_RUNTIME = "dynamic_runtime" # Mode E -- inject repeatedly on a fixed interval throughout
                                         # the call.
    CACHE_AWARE = "cache_aware"         # Mode F -- same TokenInjector primitive as C/D/E, benchmarked
                                         # against a naive "reset and replay" baseline to quantify the
                                         # cost of *not* preserving the live RingKVCache.


# Modes whose intended policy is "inject right after the user stops talking" -- these are the
# modes that benefit from (but, per RAGConfig.validate(), do not strictly require) turn-boundary
# detection from rag.turn_detector.
_MODES_USING_TURN_DETECTION = {InjectionMode.TURN_INJECTION}

_KNOWN_VECTOR_DBS = ("faiss", "chroma")
_KNOWN_EMBEDDING_MODELS = ("bge-small", "bge-base", "bge-large", "e5-small", "e5-base", "e5-large",
                           "sentence-transformers")


@dataclass
class RAGConfig:
    """One object capturing every knob the notebook / benchmark harness needs to set.

    Defaults reproduce baseline PersonaPlex behavior (enable_rag=False), so
    `RAGConfig()` is always a safe, no-op default.
    """

    enable_rag: bool = False
    injection_mode: InjectionMode = InjectionMode.BASELINE
    top_k: int = 5
    embedding_model: str = "bge-small"
    vector_db: str = "faiss"
    benchmark_mode: bool = False

    # Modes D/E specific knobs.
    vad_enabled: bool = False
    dynamic_injection_interval_s: float = 30.0  # only consulted by DYNAMIC_RUNTIME
    # Mode D re-injects on every detected turn boundary, so its per-injection token count must
    # stay small relative to `top_k` -- Mode C's own benchmark showed ~25ms/injected token, so a
    # 5-document block (340 tokens, as used by B/C) costs ~8.5s per injection, far too slow to
    # repeat every time the user pauses. Deliberately defaults much smaller than `top_k`.
    turn_injection_top_k: int = 2
    # Mode E re-injects on a fixed wall-clock interval regardless of conversational state, so the
    # same per-injection-token-count-must-stay-small reasoning as turn_injection_top_k applies --
    # kept as a separate knob (rather than reusing turn_injection_top_k) because the two modes
    # don't have to use the same retrieval breadth, even though both default to 2.
    dynamic_injection_top_k: int = 2

    # Retrieval-layer knobs (consumed by rag.retriever once implemented).
    score_threshold: float | None = None

    # Fallback query used by a live moshi.server connection when the client supplies no
    # `rag_query` of its own -- the normal case for the browser web UI, which has no way to send
    # one (PersonaPlex has no ASR, and the UI predates this project's `rag_query` parameter). When
    # empty (the default), RAGSession falls back further still to injecting the whole knowledge
    # base (Retriever.retrieve_all, capped only by `full_kb_max_chunks` below) rather than skipping
    # injection entirely -- see docs/PRODUCTION_RAG.md. Setting this to a short description of the
    # deployment's domain (e.g. "drone rental policies") lets an operator get real similarity-search
    # retrieval by default instead of the cruder "inject everything" fallback.
    default_query: str = ""

    # Cap on how many chunks the empty-query "inject everything" fallback above will use.
    # Deliberately NOT the same knob as `top_k`: `top_k` bounds a *ranked* similarity-search
    # result, where cutting off the lowest-ranked results is a reasonable tradeoff; the no-query
    # fallback has no ranking at all (chunks come back in plain insertion/file order), so capping
    # it at the same small default as `top_k` silently and deterministically drops whichever
    # chunks happen to come later in the source document -- exactly the real bug this knob fixes
    # (see docs/PRODUCTION_RAG.md Section 9). `None` (the default) means "inject the entire
    # knowledge base, uncapped" -- correct unless the knowledge base is large enough that the
    # injection latency (~25ms/token, per Mode C's benchmark) becomes a real problem, in which
    # case set this explicitly.
    full_kb_max_chunks: int | None = None

    # ---- Token-budget guard (see docs/PRODUCTION_RAG.md's root-cause writeup) ------------------
    # `full_kb_max_chunks` above caps by *chunk count*, which has no relationship to how many
    # forced-token frames a knowledge block actually costs once injected. The live model's
    # attention KV-cache (`RingKVCache`) has a FIXED capacity (`LMModel.context`, e.g. 3000 frames
    # for the released PersonaPlex checkpoint) shared by the persona prompt, the voice prompt, the
    # injected RAG knowledge, AND the live conversation that follows -- it is a ring buffer, so
    # once full, injecting more tokens silently evicts the OLDEST ones (which, at connection start,
    # means the persona prompt and the earliest knowledge chunks) before the user has even spoken.
    # Without a real token-budget check, a knowledge base whose total token count approaches the
    # model's context size (a ~12K-character document is already ~3000 tokens, i.e. the entire
    # budget) can silently overflow the attention window on every single connection. See
    # `RAGSession._compute_injection_token_budget`/`_select_within_budget`, which consult this pair
    # of knobs before any injection (every mode, not just the no-query fallback).
    #
    # Hard override for the number of tokens any single injection burst may use. `None` (the
    # default) means "compute it live" -- ask the connection's actual `RingKVCache` how many frames
    # are already used (persona/voice prompt) and how many it can hold in total, and budget the
    # difference (minus `injection_reserve_frames`). Set this explicitly only to force a smaller,
    # deterministic cap regardless of live cache state (e.g. for benchmark reproducibility).
    max_injection_tokens: int | None = None

    # Frames deliberately left unused after injection, reserved for the live conversation that
    # follows (the same ring buffer keeps filling once opus_loop starts) -- injecting all the way
    # to 100% fill means the very next user utterance starts evicting the knowledge just injected.
    #
    # Default is 100 frames (~8s @ 12.5Hz), deliberately small. An earlier default of 400 frames
    # (~32s) turned out to actively break correctness for knowledge bases sized close to the
    # model's context window: measured directly against this project's ~12,264-character
    # RobotBulls `text.txt` (21 chunks, ~2,400-2,550 tokens including the scope instruction, per
    # both a BERT-wordpiece and a GPT-2-BPE tokenizer estimate -- see docs/PRODUCTION_RAG.md),
    # subtracting 400 reserve frames from a 3000-frame context left too little budget to fit the
    # WHOLE document, so `RAGSession._select_within_budget` silently dropped the lowest-ranked
    # chunks -- whichever topics happened to score worst against the connection-start
    # `default_query` (e.g. "BTC Bull"/"Solana Bull" in that real run), making the assistant
    # unable to answer questions about exactly those topics even though they're in the document.
    # Prefer a SMALL reserve and let as much of the knowledge base fit as possible; only raise
    # this if you've confirmed (e.g. via the notebook's Section 12 verification cell) that your
    # knowledge base comfortably fits already and you specifically need headroom for very long
    # calls.
    injection_reserve_frames: int = 100

    # ---- Scope enforcement (answer only from the knowledge base) -------------------------------
    # Without this, injecting retrieved facts (or nothing, when retrieval finds nothing relevant)
    # only ever *adds* context -- it never tells the model NOT to fall back on its own pretrained
    # knowledge, so a question the knowledge base doesn't cover gets answered anyway, from
    # whatever the model already knows. `strict_scope=True` (the default) fixes this two ways:
    #   1. Every injected knowledge block is wrapped with an explicit "answer ONLY from this"
    #      instruction (see `rag.injection_manager.build_scoped_knowledge_block`).
    #   2. When retrieval finds nothing relevant (an explicit query scored below
    #      `score_threshold` against every chunk, or the knowledge base is empty), an explicit
    #      decline instruction is injected INSTEAD of silently injecting nothing -- the previous
    #      behavior left the model with no instruction at all for that case, free to answer from
    #      its own knowledge. Set to `False` to restore the old "just inject whatever was
    #      retrieved, or nothing" behavior (e.g. for Mode B's negative-control benchmark, which
    #      intentionally compares grounding quality with and without explicit scope wording).
    strict_scope: bool = True

    # Exact phrase the model should fall back to for anything the knowledge base doesn't cover.
    # Keep this short and unambiguous -- it is quoted verbatim in the instruction text the model
    # is told to repeat back, not paraphrased. Set this to something specific to your deployment
    # (e.g. naming your company/product) for a more natural-sounding decline.
    refusal_message: str = "I can only answer questions based on the provided knowledge base."

    # Where per-request logs (Phase 9) and benchmark reports (Phase 8) get written.
    log_dir: str = "rag_logs"

    def validate(self) -> list[str]:
        """Returns human-readable warnings; never raises. Designed to be called from a notebook
        cell and printed, rather than crashing a `Run All` over a config typo."""
        warnings: list[str] = []

        if not self.enable_rag and self.injection_mode != InjectionMode.BASELINE:
            warnings.append(
                "ENABLE_RAG is False but INJECTION_MODE is "
                f"'{self.injection_mode.value}' (!= 'baseline'); INJECTION_MODE will be ignored "
                "until ENABLE_RAG=True."
            )

        if self.injection_mode in _MODES_USING_TURN_DETECTION and not self.vad_enabled:
            warnings.append(
                f"INJECTION_MODE='{self.injection_mode.value}' is designed to trigger on detected "
                "end-of-turn boundaries, but VAD_ENABLED=False. With no boundary signal, this mode "
                "will never fire -- either set VAD_ENABLED=True or switch to 'dynamic_runtime' "
                "(fixed-interval injection)."
            )

        if self.top_k <= 0:
            warnings.append(f"TOP_K should be a positive integer, got {self.top_k}.")

        if self.turn_injection_top_k <= 0:
            warnings.append(
                f"turn_injection_top_k should be a positive integer, got {self.turn_injection_top_k}."
            )
        if self.injection_mode == InjectionMode.TURN_INJECTION and self.turn_injection_top_k > 3:
            warnings.append(
                f"turn_injection_top_k={self.turn_injection_top_k} is large for a per-turn "
                "re-injection -- Mode C's benchmark measured ~25ms per injected token, so "
                "5 documents (~340 tokens) cost ~8.5s per injection. Consider keeping this small "
                "(1-2) so repeated mid-conversation injections don't stall the live audio."
            )

        if self.vector_db not in _KNOWN_VECTOR_DBS:
            warnings.append(
                f"Unknown VECTOR_DB '{self.vector_db}'; expected one of {_KNOWN_VECTOR_DBS}."
            )

        if self.embedding_model not in _KNOWN_EMBEDDING_MODELS:
            warnings.append(
                f"Unrecognized EMBEDDING_MODEL '{self.embedding_model}'; expected one of "
                f"{_KNOWN_EMBEDDING_MODELS}. It may still work if it's a valid model name/path for "
                "the configured embedding backend, but double-check for typos."
            )

        if self.dynamic_injection_interval_s <= 0:
            warnings.append(
                "DYNAMIC_INJECTION_INTERVAL_S should be positive, got "
                f"{self.dynamic_injection_interval_s}."
            )

        if self.dynamic_injection_top_k <= 0:
            warnings.append(
                f"dynamic_injection_top_k should be a positive integer, got "
                f"{self.dynamic_injection_top_k}."
            )
        if self.injection_mode == InjectionMode.DYNAMIC_RUNTIME and self.dynamic_injection_top_k > 3:
            warnings.append(
                f"dynamic_injection_top_k={self.dynamic_injection_top_k} is large for a repeated "
                "fixed-interval re-injection -- Mode C's benchmark measured ~25ms per injected "
                "token. Consider keeping this small (1-2) so repeated injections don't stall the "
                "live audio."
            )

        if self.full_kb_max_chunks is not None and self.full_kb_max_chunks <= 0:
            warnings.append(
                f"full_kb_max_chunks should be a positive integer or None, got {self.full_kb_max_chunks}."
            )

        if self.max_injection_tokens is not None and self.max_injection_tokens <= 0:
            warnings.append(
                f"max_injection_tokens should be a positive integer or None, got {self.max_injection_tokens}."
            )

        if self.injection_reserve_frames < 0:
            warnings.append(
                f"injection_reserve_frames should be >= 0, got {self.injection_reserve_frames}."
            )

        if self.strict_scope and not self.refusal_message.strip():
            warnings.append(
                "strict_scope is True but refusal_message is empty -- the model would be told to "
                "respond with an empty string for out-of-scope questions. Set REFUSAL_MESSAGE to "
                "a real phrase."
            )

        return warnings

    def as_dict(self) -> dict:
        d = asdict(self)
        d["injection_mode"] = self.injection_mode.value
        return d

    def describe(self) -> str:
        """Human-readable summary, e.g. for printing at the top of a notebook cell or a log file."""
        lines = [f"{key} = {value!r}" for key, value in self.as_dict().items()]
        warnings = self.validate()
        if warnings:
            lines.append("")
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    @classmethod
    def from_env(cls, prefix: str = "PERSONAPLEX_RAG_") -> "RAGConfig":
        """Build a config from environment variables, e.g. PERSONAPLEX_RAG_ENABLE_RAG=1. Useful for
        driving the same config from a notebook cell (via os.environ) or a shell script identically."""

        def _get(name: str, default, cast=str):
            raw = os.environ.get(prefix + name)
            if raw is None:
                return default
            if cast is bool:
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return cast(raw)

        return cls(
            enable_rag=_get("ENABLE_RAG", False, bool),
            injection_mode=InjectionMode(_get("INJECTION_MODE", InjectionMode.BASELINE.value)),
            top_k=_get("TOP_K", 5, int),
            embedding_model=_get("EMBEDDING_MODEL", "bge-small"),
            vector_db=_get("VECTOR_DB", "faiss"),
            benchmark_mode=_get("BENCHMARK_MODE", False, bool),
            vad_enabled=_get("VAD_ENABLED", False, bool),
            dynamic_injection_interval_s=_get("DYNAMIC_INJECTION_INTERVAL_S", 30.0, float),
            dynamic_injection_top_k=_get("DYNAMIC_INJECTION_TOP_K", 2, int),
            log_dir=_get("LOG_DIR", "rag_logs"),
            max_injection_tokens=_get("MAX_INJECTION_TOKENS", None, int),
            injection_reserve_frames=_get("INJECTION_RESERVE_FRAMES", 100, int),
            strict_scope=_get("STRICT_SCOPE", True, bool),
            refusal_message=_get(
                "REFUSAL_MESSAGE", "I can only answer questions based on the provided knowledge base."
            ),
        )
