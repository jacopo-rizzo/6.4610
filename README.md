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

# Tests (no GPU or downloads needed)
pytest tests/
```

Any tokenizer can then be loaded with
`superspec.tokenization.load_vocab("tokenizer/<name>.yaml")`.
