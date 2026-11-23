"""Distillation step 1 (run once): the target answers UltraChat prompts.

    python scripts/train/distill_generate.py                  # vLLM if installed, else HF
    python scripts/train/distill_generate.py generate.prompts.num_prompts=1000   # quick test
    python scripts/train/distill_generate.py target_profile=targets/olmo2.yaml

Writes one JSON line per prompt to generate.output_path:
    {"index", "prompt", "prompt_ids", "response_ids", "finished"}
with ids in the target's vocabulary. `response_ids` ends with the end-of-turn
token (Llama-3 <|eot_id|>) when the target finished its answer. Re-running resumes.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from superspec.config import load_config
from superspec.eval.common import chat_prompt_ids, stop_token_ids
from superspec.tokenization import load_vocab

logger = logging.getLogger("distill_generate")


def load_prompts(prompt_cfg) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(prompt_cfg.hf_name, split=prompt_cfg.split)
    n = min(prompt_cfg.num_prompts, len(ds))
    # ultrachat_200k has the first user turn in "prompt" and the conversation in "messages".
    return [ds[i]["prompt"] for i in range(n)]


def load_extra_prompts(sources) -> list[tuple[str, str]]:
    """(source name, prompt) for each entry of generate.extra_prompts: task-like prompts
    built from a template over the first `num_prompts` rows of a dataset's TRAIN split
    (never the benchmark prompts). `{a.b}` in a template reads row["a"]["b"]; fields are
    cut to `max_chars`."""
    import re

    from datasets import load_dataset

    out = []
    for src in sources or []:
        ds = load_dataset(src.hf_name, name=src.get("subset"), split=src.split, streaming=True)
        limit = src.get("max_chars", 4000)

        def field(match, row=None):
            value = row
            for key in match.group(1).split("."):
                value = value[key]
            return str(value)[:limit]

        n = 0
        for row in ds:
            text = re.sub(r"\{([\w.]+)\}", lambda m, row=row: field(m, row), src.template).strip()
            if text:
                out.append((src.name, text))
                n += 1
            if n >= src.num_prompts:
                break
        logger.info("%s: %d prompts from %s", src.name, n, src.hf_name)
    return out


def clean_response(token_ids, finished: bool, stop_ids: set[int], eot_id: int) -> list[int]:
    """Drop any stop tokens the backend left at the end, then mark a finished answer
    with the end-of-turn token so the drafts learn where the target stops."""
    ids = list(token_ids)
    while ids and ids[-1] in stop_ids:
        ids.pop()
    return ids + [eot_id] if finished else ids


def generate_vllm(cfg, target_name: str, prompt_ids: list[list[int]], chunk_size: int = 4096):
    from vllm import LLM, SamplingParams

    llm = LLM(model=target_name, dtype="bfloat16", seed=cfg.seed)
    params = SamplingParams(temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
                            max_tokens=cfg.sampling.max_new_tokens, seed=cfg.seed)
    # Generate in chunks so finished results are written (and resumable) as we go.
    for start in range(0, len(prompt_ids), chunk_size):
        chunk = prompt_ids[start:start + chunk_size]
        for out in llm.generate([{"prompt_token_ids": ids} for ids in chunk], params):
            completion = out.outputs[0]
            yield completion.token_ids, completion.finish_reason == "stop"


def generate_hf(cfg, target_name: str, prompt_ids: list[list[int]], vocab, stop_ids, pad_id: int,
                batch_size: int = 32):
    from superspec.models import load_causal_lm

    model = load_causal_lm(target_name)
    torch.manual_seed(cfg.seed)
    for start in range(0, len(prompt_ids), batch_size):
        batch = prompt_ids[start:start + batch_size]
        width = max(map(len, batch))
        # Left padding, so every prompt ends right where generation starts.
        ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in batch], device=model.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch],
                            device=model.device)
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=True,
                             temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
                             max_new_tokens=cfg.sampling.max_new_tokens,
                             eos_token_id=sorted(stop_ids), pad_token_id=pad_id)
        for row in out[:, width:].tolist():
            ends = [i for i, t in enumerate(row) if t in stop_ids]
            yield (row[: ends[0] + 1] if ends else row), bool(ends)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/distill.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    full_cfg = load_config(args.config, args.overrides)
    cfg, target = full_cfg.generate, full_cfg.target
    tokenizer_cfg = load_config(target.tokenizer)
    vocab = load_vocab(tokenizer_cfg)
    stop_ids = stop_token_ids(vocab, tokenizer_cfg)
    special = tokenizer_cfg.special_tokens
    eot_id = vocab.hf.convert_tokens_to_ids(special.end_of_turn)
    pad_id = vocab.hf.convert_tokens_to_ids(special.pad)
    logger.info("Target %s", target.hf_name)

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0

    # Extra (task-like) prompts follow the main ones, so adding them to a finished file
    # only generates the new rows.
    tagged = [("main", p) for p in load_prompts(cfg.prompts)] + load_extra_prompts(
        cfg.get("extra_prompts"))
    sources = [s for s, _ in tagged][done:]
    prompts = [p for _, p in tagged][done:]
    logger.info("%d prompts to generate (%d already done)", len(prompts), done)
    if not prompts:
        return
    prompt_ids = [chat_prompt_ids(vocab, p) for p in prompts]

    backend = cfg.backend
    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            logger.warning("vLLM not installed, falling back to HF generate (much slower)")
            backend = "hf"
    responses = (generate_vllm(cfg, target.hf_name, prompt_ids) if backend == "vllm"
                 else generate_hf(cfg, target.hf_name, prompt_ids, vocab, stop_ids, pad_id))

    with open(out_path, "a") as f:
        for k, (token_ids, finished) in enumerate(responses):
            row = {"index": done + k, "source": sources[k], "prompt": prompts[k],
                   "prompt_ids": prompt_ids[k],
                   "response_ids": clean_response(token_ids, finished, stop_ids, eot_id),
                   "finished": finished}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 1000 == 0:
                logger.info("%d / %d", k, len(prompts))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
