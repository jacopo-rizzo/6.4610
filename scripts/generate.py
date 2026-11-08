"""Try a decoder on one prompt and print the text and speed stats.

    # baseline
    python scripts/generate.py --method autoregressive --prompt "Explain speculative decoding."

    # SuperBPE draft
    python scripts/generate.py --method superword \
        --draft $CKPT_DIR/drafts/distill/draft_50m_llama3_superbpe \
        --draft-tokenizer tokenizer/llama3_superbpe.yaml --temperature 0.7
"""

import argparse
import json

import torch

from superspec.config import load_config, load_env
from superspec.decoding import CachedLM, SamplingConfig, SpeculativeDecoder, autoregressive_generate
from superspec.eval.common import chat_prompt_ids, stop_token_ids
from superspec.models import load_causal_lm
from superspec.tokenization import load_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="superword",
                        choices=["autoregressive", "standard", "superword", "exact_match"])
    parser.add_argument("--prompt", default="Explain speculative decoding in two paragraphs.")
    parser.add_argument("--target-tokenizer", default="tokenizer/llama3.yaml")
    parser.add_argument("--target", default=None, help="defaults to the tokenizer's hf_name")
    parser.add_argument("--draft", default=None, help="draft model checkpoint")
    parser.add_argument("--draft-tokenizer", default="tokenizer/llama3.yaml")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--draft-bytes", type=int, default=32)
    parser.add_argument("--no-retokenize", action="store_true",
                        help="exact_match: use superword expansions instead of re-encoding text")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    load_env()

    target_cfg = load_config(args.target_tokenizer)
    target_vocab = load_vocab(target_cfg)
    stop_ids = stop_token_ids(target_vocab, target_cfg)
    prompt = chat_prompt_ids(target_vocab, args.prompt)
    sampling = SamplingConfig(temperature=args.temperature, top_p=args.top_p)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    target = CachedLM(load_causal_lm(args.target or target_cfg.hf_name, device=args.device))

    if args.method == "autoregressive":
        result = autoregressive_generate(target, target_vocab, prompt, args.max_new_tokens,
                                         sampling, stop_ids, generator)
    else:
        if args.draft is None:
            parser.error("--draft is required for speculative methods")
        draft_vocab = load_vocab(args.draft_tokenizer)
        draft = CachedLM(load_causal_lm(args.draft, device=args.device))
        decoder = SpeculativeDecoder(target, target_vocab, draft, draft_vocab, args.method,
                                     sampling, stop_ids, max_draft_bytes=args.draft_bytes,
                                     retokenize=not args.no_retokenize)
        result = decoder.generate(prompt, args.max_new_tokens, generator)

    print(target_vocab.decode(result.tokens))
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
