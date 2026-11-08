# 6.4610
Speculative Decoding with Superword Tokenization

## Setup

```bash
pip install torch          # the build matching your CUDA version
pip install -e ".[dev]"
cp .env.example .env       # fill in HF_TOKEN, HF_HOME, CKPT_DIR
```

## Pipeline so far

```bash
# 1. One-time: learn the SuperBPE draft tokenizer (Llama-3 vocab + 30k superwords)
python scripts/train_superbpe.py --config tokenizer/llama3_superbpe.yaml

# 2. Pretokenize the pretraining mix, once per tokenizer
python scripts/pretokenize.py --tokenizer tokenizer/llama3.yaml
python scripts/pretokenize.py --tokenizer tokenizer/llama3_superbpe.yaml

# 3. Pretrain each draft (model x tokenizer), then distill it on target responses
python scripts/pretrain.py --model model/draft_50m.yaml --tokenizer tokenizer/llama3.yaml
python scripts/pretrain.py --model model/draft_50m.yaml --tokenizer tokenizer/llama3_superbpe.yaml
python scripts/distill_generate.py        # once: target answers UltraChat prompts
python scripts/distill_train.py train.model=model/draft_50m.yaml train.tokenizer=tokenizer/llama3.yaml
python scripts/distill_train.py train.model=model/draft_50m.yaml train.tokenizer=tokenizer/llama3_superbpe.yaml
# (multi-GPU: `accelerate launch scripts/pretrain.py ...`; re-running resumes)

# Try a decoder on one prompt (autoregressive | standard | superword | exact_match)
python scripts/generate.py --method autoregressive --prompt "Explain speculative decoding."

# 4. Evaluate (settings in configs/eval/; results go to $RESULTS_DIR)
python scripts/eval_heldout.py                                  # draft loss, bits/byte
python scripts/eval_speed.py --config eval/specbench.yaml       # speed + acceptance
python scripts/eval_speed.py --config eval/humaneval.yaml
python scripts/eval_speed.py --config eval/mbpp.yaml
python scripts/eval_lossless.py --config eval/specbench.yaml    # sampled outputs match target?

# Tests (no GPU or downloads needed; test_decoding.py needs torch + transformers, ~3 min on CPU)
pytest tests/
```

Any tokenizer can then be loaded with
`superspec.tokenization.load_vocab("tokenizer/<name>.yaml")`.
