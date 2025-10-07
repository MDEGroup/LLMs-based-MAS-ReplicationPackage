# Developing LLM-based Multi-Agent Systems in Software Engineering: A Mixed-Method Experience Report




## Overview

This repository is the replication package for the experience report "Developing LLM-based Multi-Agent Systems in Software Engineering: A Mixed-Method Experience Report" (De Oliveira et al., 2025) submitted to Empirical Software Engineering (EMSE) journal for publication. The work presents a comparative and empirical study of frameworks that orchestrate large language models (LLMs) via multi-agent systems (MAS). The replication package contains code, prompts, datasets, and analysis scripts used to evaluate framework coverage, developer-oriented characteristics, and practical performance in a README summarization use case.


### Authors

Mariama Celi Serafim De Oliveira, Motunrayo Osatohanmen Ibiyo, Marco Gianrusso, Claudio Di Sipio, Davide Di Ruscio, Phuong T. Nguyen  
University of L’Aquila, Via Vetoio, L’Aquila, 67100, Italy



## Repository structure

This repository contains the materials used for the README summarization experiments and analysis with different MAS frameworks
  - `analysis_results/` — Notebooks and scripts used to analyze results and generate plots. In particular:
    - `evaluation/` — it contains evaluation outputs in CSV format  
    - `token_usage/` — Token consumption logs for different frameworks and experimental runs.


For each tested MAS frameworks, we report the prompt files and tuned/optimized prompts used in the experiments 
- `autogen/`, `autogpt/`, `dify/`, `semantic_kernel/`, `semantic_kernel_chat/`, `haystack/`, `llama-index/` contains the implementation for each corresponding framework


- `results/` folder contains with evaluation CSVs and selected best prompts.









