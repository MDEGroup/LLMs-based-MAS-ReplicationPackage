# Developing LLM-based Multi-Agent Systems in Software Engineering: A Mixed-Method Experience Report



## Authors

Mariama Celi Serafim De Oliveira, Motunrayo Osatohanmen Ibiyo, Marco Gianrusso, Claudio Di Sipio, Davide Di Ruscio, Phuong T. Nguyen  
University of L’Aquila, Via Vetoio, L’Aquila, 67100, Italy

## Overview

This repository is the replication package for the experience report "Developing LLM-based Multi-Agent Systems in Software Engineering: A Mixed-Method Experience Report" (De Oliveira et al., 2025) submitted to Empirical Software Engineering (EMSE) journal for publication. The work presents a comparative and empirical study of frameworks that orchestrate large language models (LLMs) via multi-agent systems (MAS). The replication package contains code, prompts, datasets, and analysis scripts used to evaluate framework coverage, developer-oriented characteristics, and practical performance in a README summarization use case.


## Repository structure (high level)

- `reproduce_summary_extractor/` — Main replication materials for the README summarization experiments and analysis.
  - `analysis_results/` — Notebooks and scripts used to analyze results and generate plots (e.g., `evaluation_results.ipynb`, `graphs.ipynb`).
  - `evaluation/` — it contains evaluation outputs in CSV format  
  - `token_usage/` — Token consumption logs for different frameworks and experimental runs.

- `autogen/`, `autogpt/`, `dify/`, `semantic_kernel/`, `semantic_kernel_chat/` — Top-level framework experiment folders. Each typically includes:
  - Experiment drivers and evaluation scripts (`main.py`, `evaluation.py`, etc.).
  - Prompt files and tuned/optimized prompts used in the experiments (`optimized_prompt.txt`, `data_prompts.txt`).
  - `results/` folders with evaluation CSVs and selected best prompts.

- `dify/rouge-score/` — Local ROUGE scoring utility used to compute evaluation metrics. Contains its own `requirements.txt` and usage notes.

Quick start

1. Choose the framework experiment you want to reproduce. Example paths:
   - `reproduce_summary_extractor/autogpt`
   - `reproduce_summary_extractor/dify`
   - `reproduce_summary_extractor/semantic_kernel`
2. Inspect the corresponding `results/` directory to review `evaluation_*.csv` outputs and `best_prompts_*.txt` files.
3. Install the Python dependencies listed in any `requirements.txt` present in the framework folder (for Windows PowerShell, use `python -m pip install -r requirements.txt`).
4. Set required API keys and environment variables as described in the top of each framework folder (these experiments often require keys for hosted LLM providers).
5. Run the provided experiment driver (for example, `python main.py` or `python evaluation.py`) following instructions inside each folder.






