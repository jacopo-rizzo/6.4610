#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline with tiny budgets (configs/smoke/).
# Everything is written under $CKPT_DIR/smoke and $RESULTS_DIR/smoke, so it
# never touches real runs. Needs one GPU; roughly 30-60 minutes.
#
#   bash scripts/smoke_test.sh               # Llama-3 target: every step, then checks
#   TARGET=olmo2 bash scripts/smoke_test.sh  # OLMo-2 target
#   SKIP_TESTS=1 bash scripts/smoke_test.sh
#   FROM_STEP=5 bash scripts/smoke_test.sh   # resume from a step (earlier outputs kept)
#
# Each step resumes or skips work that is already done, so re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi
: "${CKPT_DIR:?Set CKPT_DIR (e.g. in .env)}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
FROM_STEP="${FROM_STEP:-0}"
TARGET="${TARGET:-llama3}"              # a profile in configs/targets/
PROFILE="target_profile=targets/$TARGET.yaml"
echo "Smoke test for target: $TARGET"

step() {   # step <number> <description>
    CURRENT=$1
    if [ "$1" -ge "$FROM_STEP" ]; then echo; echo "=== [$1] $2 ==="; return 0; fi
    return 1
}

if [ -z "${SKIP_TESTS:-}" ] && step 0 "unit tests"; then
    pytest -q tests/
fi

if step 1 "learn the SuperBPE tokenizer (2k merges on 50 MB)"; then
    python scripts/train/train_superbpe.py --config "smoke/${TARGET}_superbpe.yaml"
fi

if step 2 "pretokenize 50 MB with both tokenizers"; then
    for tok in standard superbpe; do
        python scripts/train/pretokenize.py --config smoke/pretrain.yaml --tokenizer $tok "$PROFILE"
    done
fi

if step 3 "target answers 200 UltraChat prompts"; then
    python scripts/train/distill_generate.py --config smoke/distill.yaml "$PROFILE"
fi

if step 4 "pretrain both 50M drafts (~350 steps each)"; then
    for tok in standard superbpe; do
        python scripts/train/pretrain.py --config smoke/pretrain.yaml --tokenizer $tok "$PROFILE"
    done
fi

if step 5 "distill both drafts"; then
    for tok in standard superbpe; do
        python scripts/train/distill_train.py --config smoke/distill.yaml --tokenizer $tok "$PROFILE"
    done
fi

if step 6 "eval scripts on a few prompts"; then
    if [ ! -f data/spec_bench/question.jsonl ]; then
        mkdir -p data/spec_bench
        curl -L -o data/spec_bench/question.jsonl \
            https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl
    fi
    python scripts/eval/heldout.py --config smoke/eval.yaml "$PROFILE"
    python scripts/eval/speed.py --config smoke/eval.yaml \
        --output-dir "$RESULTS_DIR/smoke/$TARGET/specbench/speed" "$PROFILE"
    python scripts/eval/lossless.py --config smoke/eval.yaml "$PROFILE"
fi

echo; echo "=== checking saved outputs ==="
python scripts/check_smoke_outputs.py "$PROFILE"
