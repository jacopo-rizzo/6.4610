# 6.4610
Speculative Decoding with Superword Tokenization

## Setup

```bash
pip install torch          # the build matching your CUDA version
pip install -e ".[dev]"
cp .env.example .env       # fill in HF_TOKEN, HF_HOME, CKPT_DIR
```

## Smoke test (start here)

Runs the whole pipeline with tiny budgets (configs/smoke/) on one GPU in
roughly 30-60 minutes, writing only under `$CKPT_DIR/smoke` and
`$RESULTS_DIR/smoke`, then checks every saved artifact:

```bash
bash scripts/smoke_test.sh                 # unit tests + all steps + checks
TARGET=olmo2 bash scripts/smoke_test.sh    # same for the OLMo-2 target
FROM_STEP=4 bash scripts/smoke_test.sh     # resume from a step after fixing something
python scripts/check_smoke_outputs.py      # just the checks
```

## Pipeline

Commands default to the Llama-3.1-8B-Instruct target. For OLMo-2-7B-Instruct,
add `target_profile=targets/olmo2.yaml` to any command (profiles live in
`configs/targets/`). `--tokenizer standard|superbpe` picks the target's own
tokenizer or its SuperBPE draft tokenizer.

```bash
# 1. One-time: learn the SuperBPE draft tokenizer (target vocab + 30k superwords)
python scripts/train/train_superbpe.py --config tokenizer/llama3_superbpe.yaml   # or olmo2_superbpe

# 2. Pretokenize the pretraining mix, once per tokenizer
python scripts/train/pretokenize.py --tokenizer standard
python scripts/train/pretokenize.py --tokenizer superbpe

# 3. Pretrain each draft (model x tokenizer), then distill it on target responses
python scripts/train/pretrain.py --model model/draft_50m.yaml --tokenizer standard
python scripts/train/pretrain.py --model model/draft_50m.yaml --tokenizer superbpe
python scripts/train/distill_generate.py        # once per target: answers to UltraChat prompts
python scripts/train/distill_train.py --model model/draft_50m.yaml --tokenizer standard
python scripts/train/distill_train.py --model model/draft_50m.yaml --tokenizer superbpe
# (multi-GPU: `accelerate launch scripts/train/pretrain.py ...`; re-running resumes)

# 4. Evaluate (settings in configs/eval/; results go to $RESULTS_DIR/<target>)
python scripts/eval/heldout.py                                  # draft loss, bits/byte
python scripts/eval/speed.py --config eval/specbench.yaml       # speed + acceptance
python scripts/eval/speed.py --config eval/humaneval.yaml
python scripts/eval/speed.py --config eval/mbpp.yaml
python scripts/eval/lossless.py --config eval/specbench.yaml    # sampled outputs match target?

# Try a decoder on one prompt (autoregressive | standard | superword | exact_match)
python scripts/eval/generate.py --method autoregressive --prompt "Explain speculative decoding."

# Tests (no GPU or downloads needed; test_decoding.py needs torch + transformers, ~3 min on CPU)
pytest tests/
```

Any tokenizer can then be loaded with
`superspec.tokenization.load_vocab("tokenizer/<name>.yaml")`.
