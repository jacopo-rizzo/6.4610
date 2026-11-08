"""Autoregressive and speculative decoding.

    target = CachedLM(load_causal_lm("meta-llama/Llama-3.1-8B-Instruct"))
    decoder = SpeculativeDecoder(target, target_vocab, CachedLM(draft_model), draft_vocab,
                                 method="superword", sampling=SamplingConfig(temperature=0.7),
                                 stop_ids=stop_ids)
    result = decoder.generate(prompt_ids, max_new_tokens=256)
"""

from superspec.decoding.autoregressive import autoregressive_generate
from superspec.decoding.kv_cache import CachedLM
from superspec.decoding.result import GenerationResult
from superspec.decoding.sampling import SamplingConfig
from superspec.decoding.speculative import CONTROL_METHODS, METHODS, SpeculativeDecoder

__all__ = ["CONTROL_METHODS", "CachedLM", "GenerationResult", "METHODS", "SamplingConfig",
           "SpeculativeDecoder",
           "autoregressive_generate"]
