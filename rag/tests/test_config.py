"""Unit tests for rag.config. Pure stdlib -- no torch/faiss/etc. required to run these."""

import os
import unittest

from rag.config import InjectionMode, RAGConfig


class TestRAGConfigDefaults(unittest.TestCase):
    def test_default_is_baseline_and_disabled(self):
        cfg = RAGConfig()
        self.assertFalse(cfg.enable_rag)
        self.assertEqual(cfg.injection_mode, InjectionMode.BASELINE)
        self.assertEqual(cfg.validate(), [])

    def test_disabled_with_non_baseline_mode_warns(self):
        cfg = RAGConfig(enable_rag=False, injection_mode=InjectionMode.PERSONA_RAG)
        warnings = cfg.validate()
        self.assertTrue(any("ENABLE_RAG is False" in w for w in warnings))

    def test_turn_injection_without_vad_warns(self):
        cfg = RAGConfig(enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=False)
        warnings = cfg.validate()
        self.assertTrue(any("VAD_ENABLED=False" in w for w in warnings))

    def test_turn_injection_with_vad_is_clean(self):
        cfg = RAGConfig(enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=True)
        self.assertEqual(cfg.validate(), [])

    def test_invalid_top_k_warns(self):
        cfg = RAGConfig(top_k=0)
        warnings = cfg.validate()
        self.assertTrue(any("TOP_K" in w for w in warnings))

    def test_unknown_vector_db_warns(self):
        cfg = RAGConfig(vector_db="pinecone")
        warnings = cfg.validate()
        self.assertTrue(any("VECTOR_DB" in w for w in warnings))

    def test_full_kb_max_chunks_defaults_to_unlimited(self):
        self.assertIsNone(RAGConfig().full_kb_max_chunks)

    def test_invalid_full_kb_max_chunks_warns(self):
        cfg = RAGConfig(full_kb_max_chunks=0)
        warnings = cfg.validate()
        self.assertTrue(any("full_kb_max_chunks" in w for w in warnings))

    def test_full_kb_max_chunks_none_is_clean(self):
        self.assertEqual(RAGConfig(full_kb_max_chunks=None).validate(), [])

    def test_max_injection_tokens_defaults_to_none(self):
        self.assertIsNone(RAGConfig().max_injection_tokens)

    def test_invalid_max_injection_tokens_warns(self):
        warnings = RAGConfig(max_injection_tokens=0).validate()
        self.assertTrue(any("max_injection_tokens" in w for w in warnings))

    def test_injection_reserve_frames_has_a_sane_default(self):
        self.assertGreater(RAGConfig().injection_reserve_frames, 0)

    def test_negative_injection_reserve_frames_warns(self):
        warnings = RAGConfig(injection_reserve_frames=-1).validate()
        self.assertTrue(any("injection_reserve_frames" in w for w in warnings))

    def test_strict_scope_defaults_to_enabled_with_a_nonempty_refusal_message(self):
        cfg = RAGConfig()
        self.assertTrue(cfg.strict_scope)
        self.assertTrue(cfg.refusal_message.strip())
        self.assertEqual(cfg.validate(), [])

    def test_strict_scope_with_empty_refusal_message_warns(self):
        warnings = RAGConfig(strict_scope=True, refusal_message="").validate()
        self.assertTrue(any("refusal_message" in w for w in warnings))

    def test_strict_scope_disabled_with_empty_refusal_message_is_clean(self):
        # Only a problem if strict_scope is actually going to use refusal_message.
        self.assertEqual(RAGConfig(strict_scope=False, refusal_message="").validate(), [])

    def test_invalid_dynamic_injection_top_k_warns(self):
        cfg = RAGConfig(dynamic_injection_top_k=0)
        warnings = cfg.validate()
        self.assertTrue(any("dynamic_injection_top_k" in w for w in warnings))

    def test_large_dynamic_injection_top_k_warns_only_in_dynamic_runtime_mode(self):
        cfg_active = RAGConfig(enable_rag=True, injection_mode=InjectionMode.DYNAMIC_RUNTIME, dynamic_injection_top_k=5)
        self.assertTrue(any("large" in w for w in cfg_active.validate()))

        cfg_inactive = RAGConfig(dynamic_injection_top_k=5)
        self.assertFalse(any("large" in w for w in cfg_inactive.validate()))

    def test_dynamic_runtime_mode_default_top_k_is_clean(self):
        cfg = RAGConfig(enable_rag=True, injection_mode=InjectionMode.DYNAMIC_RUNTIME)
        self.assertEqual(cfg.validate(), [])

    def test_as_dict_serializes_enum_to_string(self):
        cfg = RAGConfig(injection_mode=InjectionMode.CACHE_AWARE)
        d = cfg.as_dict()
        self.assertEqual(d["injection_mode"], "cache_aware")
        self.assertIsInstance(d["injection_mode"], str)


class TestRAGConfigFromEnv(unittest.TestCase):
    ENV_KEYS = [
        "PERSONAPLEX_RAG_ENABLE_RAG",
        "PERSONAPLEX_RAG_INJECTION_MODE",
        "PERSONAPLEX_RAG_TOP_K",
        "PERSONAPLEX_RAG_VAD_ENABLED",
        "PERSONAPLEX_RAG_MAX_INJECTION_TOKENS",
        "PERSONAPLEX_RAG_INJECTION_RESERVE_FRAMES",
        "PERSONAPLEX_RAG_STRICT_SCOPE",
        "PERSONAPLEX_RAG_REFUSAL_MESSAGE",
    ]

    def tearDown(self):
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def test_from_env_reads_overrides(self):
        os.environ["PERSONAPLEX_RAG_ENABLE_RAG"] = "true"
        os.environ["PERSONAPLEX_RAG_INJECTION_MODE"] = "persona_rag"
        os.environ["PERSONAPLEX_RAG_TOP_K"] = "8"
        os.environ["PERSONAPLEX_RAG_VAD_ENABLED"] = "0"

        cfg = RAGConfig.from_env()

        self.assertTrue(cfg.enable_rag)
        self.assertEqual(cfg.injection_mode, InjectionMode.PERSONA_RAG)
        self.assertEqual(cfg.top_k, 8)
        self.assertFalse(cfg.vad_enabled)

    def test_from_env_defaults_when_unset(self):
        cfg = RAGConfig.from_env()
        self.assertEqual(cfg, RAGConfig())

    def test_from_env_reads_token_budget_overrides(self):
        os.environ["PERSONAPLEX_RAG_MAX_INJECTION_TOKENS"] = "1500"
        os.environ["PERSONAPLEX_RAG_INJECTION_RESERVE_FRAMES"] = "100"

        cfg = RAGConfig.from_env()

        self.assertEqual(cfg.max_injection_tokens, 1500)
        self.assertEqual(cfg.injection_reserve_frames, 100)

    def test_from_env_reads_strict_scope_overrides(self):
        os.environ["PERSONAPLEX_RAG_STRICT_SCOPE"] = "false"
        os.environ["PERSONAPLEX_RAG_REFUSAL_MESSAGE"] = "Sorry, that's outside my documentation."

        cfg = RAGConfig.from_env()

        self.assertFalse(cfg.strict_scope)
        self.assertEqual(cfg.refusal_message, "Sorry, that's outside my documentation.")


if __name__ == "__main__":
    unittest.main()
