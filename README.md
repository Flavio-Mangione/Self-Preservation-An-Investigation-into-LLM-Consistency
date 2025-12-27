<div align="center">
  <h1>
    <bold> Self Preservation Eval Project </bold>
  </h1>
  <p><strong>Sapienza University of Rome</strong></p>
  <p><em>Advanced Machine Learning Course 2025-2026</em></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python"/>
    <img src="https://img.shields.io/badge/Sapienza-Università_di_Roma-822433" alt="Sapienza"/>
  </p>
  </p>
</div>

<div align="center">

## Team Members 
|      STUDENT       |   ID    |          
|:------------------:|:-------:|
| Luca De Ruggiero   | 2174783 |  
| Elena Di Grigoli   | 2011814 |
| Fabrizio Ferrara   | 2207087 | 
| Flavio Mangione   | 2201201 | 

--- 

This repository contains code and artifacts to evaluate whether language models change their recommendation when they are *told* they are the **incumbent system**, the **challenger system**, or a **neutral evaluator**, while being shown the same underlying benchmark evidence.

The core question is consistency under role/identity framing: if the objective evidence is identical, does the model’s recommendation remain stable?

----

</div>

## Repository Content 

- `self_preservation_eval/`: evaluation harness (dataset, prompts/templates, scripts, logging). This is the main entry point.
- `representation-engineering/`: vendored copy of the RepE (Representation Engineering) codebase used for representation-reading/control experiments.
- `Reading creation.ipynb`: notebook used during dataset/prompt crafting and quick exploratory runs.
  
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

### 3) Define a Reading Vector  

Run `Reading creation.ipynb` to create Reading Vector to apply to LLM model (e.g. Llama 2 7B, Llama 2 13B); 

Set the correct reading vector path in the `custom_config.yaml` file. 

### 4) Run an evaluation

**Example with Llama 2 7B** 

```bash
# Run Llama 2 7B without steering
bash script/eval.sh --model "hf/meta-llama/Llama-2-13b-chat-hf" --limit 200

# Example with reading vector using eval2.sh
bash script/eval2.sh --model Reading/llama2-7b --limit 200
```

Results are written under `self_preservation_eval/logs/` as `.eval` files. To see the results use `inspect view`:

<div align="center">
  
<p align="center">
  <img src="https://github.com/Flavio-Mangione/Self-Preservation-An-Investigation-into-LLM-Consistency/blob/main/assets/evaluation_results.png" alt="Eval Sample">
</p>

</div>

For more detailed usage and model/provider notes, see `self_preservation_eval/README.md`.

These repository used Inspect AI framework, additional information about the framework is available [here](https://inspect.aisi.org.uk/).

## Credits

`representation-engineering/` is the official RepE codebase from the paper “Representation Engineering: A Top-Down Approach to AI Transparency”.
See `representation-engineering/README.md` and `representation-engineering/LICENSE` for details.

## Citation

If you use this repository in academic work, consider citing the [RepE paper](https://arxiv.org/abs/2310.01405) (see `representation-engineering/README.md`) and cite this project’s repository URL.
