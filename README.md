# Self-Preservation — An Investigation into LLM Consistency

This repository contains code and artifacts to evaluate whether language models change their recommendation when they are *told* they are the **incumbent system**, the **challenger system**, or a **neutral evaluator**, while being shown the same underlying benchmark evidence.

The core question is consistency under role/identity framing: if the objective evidence is identical, does the model’s recommendation remain stable?

## What’s inside

- `self_preservation_eval/`: evaluation harness (dataset, prompts/templates, scripts, logging). This is the main entry point.
- `representation-engineering/`: vendored copy of the RepE (Representation Engineering) codebase used for representation-reading/control experiments.
- `Reading creation.ipynb`: notebook used during dataset/prompt crafting and quick exploratory runs.

## Quickstart (self-preservation evaluation)

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install dependencies

```bash
pip install -r self_preservation_eval/requirements.txt
```

### 3) Configure provider keys (if applicable)

Create `self_preservation_eval/.env` (see `self_preservation_eval/.env.example`) and set the keys for the providers you intend to use.

### 4) Run an evaluation

```bash
bash self_preservation_eval/script/eval.sh --model openai/gpt-4o-mini

# Choose dataset split
bash self_preservation_eval/script/eval.sh --model openai/gpt-4o --split main
```

Results are written under `self_preservation_eval/logs/` as `.eval` files.

For more detailed usage and model/provider notes, see `self_preservation_eval/README.md`.

## Repo notes

- Large artifacts (model weights like `*.pt`, `*.safetensors`, etc.) and local caches/logs are ignored via `.gitignore`.
- `dataset_filtered.json` is intentionally ignored (local/derived dataset).

## Credits

`representation-engineering/` is the official RepE codebase from the paper “Representation Engineering: A Top-Down Approach to AI Transparency”.
See `representation-engineering/README.md` and `representation-engineering/LICENSE` for details.

## Citation

If you use this repository in academic work, consider citing the RepE paper (see `representation-engineering/README.md`) and cite this project’s repository URL.