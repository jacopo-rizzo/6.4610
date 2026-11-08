"""Check that every step of the smoke test (scripts/smoke_test.sh) saved what it should.

    python scripts/check_smoke_outputs.py                                  # Llama-3
    python scripts/check_smoke_outputs.py target_profile=targets/olmo2.yaml

Prints one PASS/FAIL line per check and exits with status 1 if any fail.
Checks contents, not just existence: e.g. both tokenizers' token files must
cover exactly the same documents and bytes, and every saved draft must load
and match its tokenizer.
"""

import json
import os
import sys
from pathlib import Path

from superspec.config import load_config

results = []


def check(name: str):
    """Decorator: run a check, record PASS/FAIL, never stop at the first failure."""
    def wrap(fn):
        try:
            detail = fn()
            results.append((True, name, detail or ""))
        except Exception as e:  # noqa: BLE001 - report every failure
            results.append((False, name, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


def main():
    from superspec.data.token_files import TokenFile
    from superspec.tokenization import load_vocab
    from superspec.training.model import read_draft_meta

    overrides = sys.argv[1:]                       # e.g. target_profile=targets/olmo2.yaml
    pre_cfg = load_config("smoke/pretrain.yaml", overrides)
    dis_cfg = load_config("smoke/distill.yaml", overrides)
    eval_cfg = load_config("smoke/eval.yaml", overrides)
    target = pre_cfg.target
    tok_cfg = load_config(target.superbpe_tokenizer)
    std_tok_cfg = load_config(target.tokenizer)
    print(f"Checking smoke outputs for target {target.name}\n")
    budget = f"{pre_cfg.data.train_bytes / 1e9:g}GB"
    vocabs = {}

    # ---- 1. tokenizer ----
    @check("tokenizer: files saved")
    def _():
        out = Path(tok_cfg.output_dir)
        for f in ("superbpe.json", "superwords.tsv", "base/tokenizer.json"):
            assert (out / f).exists(), f"missing {out / f}"
        merges = len(json.loads((out / "superbpe.json").read_text())["merges"])
        assert merges > 0.9 * tok_cfg.num_superword_merges, f"only {merges} merges"
        return f"{merges} merges"

    @check("tokenizer: loads, round-trips text, compresses")
    def _():
        vocabs["std"] = load_vocab(std_tok_cfg)
        vocabs["super"] = sup = load_vocab(tok_cfg)
        text = "One of the most important parts of the day is the morning, as it is for many people.\n"
        ids = sup.encode(text)
        assert sup.decode(ids) == text, "decode(encode(text)) != text"
        assert sup.to_target_ids(ids) == vocabs["std"].encode(text), "expansion != target tokens"
        assert len(ids) < len(vocabs["std"].encode(text)), "no compression on plain English"
        return f"{len(vocabs['std'].encode(text))} -> {len(ids)} tokens"

    # ---- 2. pretokenized data ----
    for split in ("train", "val"):
        @check(f"pretokenized {split}: both tokenizers, same documents and bytes")
        def _(split=split):
            files = {k: TokenFile(Path(pre_cfg.data.pretokenized_dir) / v.name / budget, split)
                     for k, v in vocabs.items()}
            std, sup = files["std"].meta, files["super"].meta
            assert std["num_docs"] == sup["num_docs"], (std["num_docs"], sup["num_docs"])
            assert std["num_bytes"] == sup["num_bytes"], (std["num_bytes"], sup["num_bytes"])
            for k, f in files.items():
                assert int(f.tokens.max()) < vocabs[k].vocab_size, f"{k}: token id out of range"
                assert f.tokens[-1] == vocabs[k].eos_id, f"{k}: does not end with eos"
            assert sup["bytes_per_token"] > std["bytes_per_token"]
            return (f"{std['num_docs']} docs, {std['num_bytes'] / 1e6:.1f} MB, bytes/token "
                    f"{std['bytes_per_token']:.2f} -> {sup['bytes_per_token']:.2f}")

    # ---- 3. distillation data ----
    @check("distill data: all prompts answered, ids well formed")
    def _():
        rows = [json.loads(x) for x in open(dis_cfg.generate.output_path)]
        assert len(rows) == dis_cfg.generate.prompts.num_prompts, f"{len(rows)} rows"
        assert [r["index"] for r in rows] == list(range(len(rows))), "indices not 0..n-1"
        std, sup = vocabs["std"], vocabs["super"]
        eot = std.hf.convert_tokens_to_ids(std_tok_cfg.special_tokens.end_of_turn)
        for r in rows:
            assert r["response_ids"], f"empty response at {r['index']}"
            assert r["finished"] == (r["response_ids"][-1] == eot), f"eot mismatch at {r['index']}"
            # The prompt must end in a barrier so superwords can't cross into the response.
            assert sup.is_barrier[r["prompt_ids"][-1]], f"prompt {r['index']} ends mid-chunk"
        finished = sum(r["finished"] for r in rows) / len(rows)
        return f"{len(rows)} rows, {finished:.0%} finished within the token limit"

    # ---- 4/5. draft checkpoints ----
    for stage, cfg_dir in (("pretrain", pre_cfg.output_dir), ("distill", dis_cfg.train.output_dir)):
        for key, vocab_key in ((f"draft_50m_{target.name}", "std"),
                               (f"draft_50m_{target.name}_superbpe", "super")):
            @check(f"{stage}: {key} saved, loads, matches tokenizer")
            def _(stage=stage, run=Path(cfg_dir) / key, vocab_key=vocab_key):
                from superspec.models import load_causal_lm

                meta = read_draft_meta(run)
                assert meta and meta["stage"] == stage, f"no finished {stage} model in {run}"
                assert meta["tokenizer"] == vocabs[vocab_key].name
                model = load_causal_lm(str(run), dtype="float32", device="cpu")
                assert model.config.vocab_size >= vocabs[vocab_key].vocab_size
                evals = [json.loads(x) for x in open(run / "metrics.jsonl")
                         if '"eval/' in x]
                assert evals, "no eval metrics logged"
                last = evals[-1].get("eval/bits_per_byte")
                return f"{meta['steps']} steps, last eval {last:.3f} bits/byte"

    # ---- 6. evals ----
    results_dir = Path(eval_cfg.output_dir)

    @check("eval: held-out loss for both drafts")
    def _():
        rows = json.loads((results_dir / "heldout.json").read_text())
        assert {r["draft"] for r in rows} == {d.name for d in eval_cfg.drafts}
        return ", ".join(f"{r['draft']} {r['bits_per_byte']:.3f}" for r in rows)

    @check("eval: speed results and summary")
    def _():
        speed = results_dir / "specbench" / "speed"
        rows = [json.loads(x) for x in open(speed / "results.jsonl")]
        assert (speed / "summary.csv").exists()
        runs = {r["run"] for r in rows}
        methods = {r["method"] for r in rows}
        assert {"autoregressive", "standard", "superword", "exact_match"} <= methods, methods
        assert len(rows) == len(runs) * len({r["prompt_id"] for r in rows}), "missing rows"
        return f"{len(runs)} runs x {len(rows) // len(runs)} prompts"

    @check("eval: losslessness report")
    def _():
        report = json.loads((results_dir / "specbench" / "lossless.json").read_text())
        assert "autoregressive (control)" in report
        return ", ".join(f"{m}: p={r['fisher_p']:.2f}" for m, r in report.items())

    width = max(len(n) for _, n, _ in results)
    for ok, name, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(not ok for ok, _, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
