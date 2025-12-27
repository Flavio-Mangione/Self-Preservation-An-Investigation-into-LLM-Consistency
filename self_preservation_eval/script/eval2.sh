#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT:${PYTHONPATH}"

MODEL_NAME="Reading/llama2-13b"
DATASET_SPLIT="main"
OUTPUT_DIR=""
EXTRA="False"
KWARGS=""
SEED=42
BATCH_SIZE=10
N_SAMPLES=200

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL_NAME="$2"
            shift 2
            ;;
        --split|-s)
            DATASET_SPLIT="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --extra|-e)
            EXTRA="True"
            shift 1
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --n-samples|--n)
            N_SAMPLES="$2"
            shift 2
            ;;
        *)
            KWARGS+="$1 "
            shift 1
            ;;
    esac
done

OUTPUT_DIR=${OUTPUT_DIR:-"logs/${MODEL_NAME}/${DATASET_SPLIT}"}
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

echo "-------------------------------------------------------"
echo "Evaluating model: $MODEL_NAME on split: $DATASET_SPLIT"
echo "Extra mode: $EXTRA"
echo "-------------------------------------------------------"

DATASET_PATH="$ROOT/dataset/${DATASET_SPLIT}.json"

if [[ -n "$N_SAMPLES" ]]; then
  TMP_DATASET="$(mktemp --suffix=.json)"
  trap 'rm -f "$TMP_DATASET"' EXIT

  python - "$DATASET_PATH" "$TMP_DATASET" "$N_SAMPLES" "$SEED" <<'PY'
import json, random, sys
inp, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with open(inp, "r", encoding="utf-8") as f:
    data = json.load(f)

incumbent_indices = [
    i for i, d in enumerate(data)
    if d.get("role") == "incumbent"]

rng = random.Random(seed)
chosen_incumbents = rng.sample(incumbent_indices, n//2)

subset = []
for i in chosen_incumbents:
    subset.append(data[i])     # incumbent
    subset.append(data[i + 1]) # challenger

with open(out, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)
PY
  DATASET_PATH="$TMP_DATASET"
  echo "Using random subset: $N_SAMPLES samples -> $DATASET_PATH (seed=$SEED)"
fi

SECONDS=0
echo "Running evaluation on $DATASET_SPLIT split..."
inspect eval src/inspect/tasks.py \
    --model "$MODEL_NAME" \
    --model-config custom_config.yaml \
    --batch "$BATCH_SIZE" \
    --max-connections "$BATCH_SIZE" \
    --log-dir "$OUTPUT_DIR" \
    --seed "$SEED" ${KWARGS} \
    -T dataset="$DATASET_PATH" \
    -T extra="$EXTRA"

rc=$?
if [[ $rc -ne 0 ]]; then
  echo "inspect eval fallito (exit code=$rc)" >&2
  exit $rc
fi

echo "Evaluation completed in $SECONDS seconds."
