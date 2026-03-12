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

## Running the Experiments

Each framework implementation is located in its corresponding directory (e.g., `semantic_kernel/`, `autogen/`, `dify/`).  
The frameworks which depend on third libraries or APIs to run follow the same setup procedure described below.

### Python Environment

All experiments were executed using **Python 3.12**.

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Then install the required dependencies:
```bash
pip install -r requirements.txt
```
Each framework folder contains its own requirements.txt file specifying the required dependencies.

### Environment Variables

Some frameworks require API credentials to access large language models.

Where applicable, an .env.example file is provided.
Create your configuration file by copying it:

```cp .env.example .env```

Then edit .env and provide the required API keys:

```OPENAI_API_KEY=your_api_key_here```

---
### Framework Implementations

#### AutoGen

The `autogen/METAGENTE` directory contains the implementation based on AutoGen.

Run the experiment:

For optimization workflow
```bash
python main.py
```

For evaluation workflow
```bash
python evaluation.py
```

---

#### AutoGPT

The implementation related to the AutoGPT framework could not be fully exported due to limitations in exporting configured agents from the platform.

To ensure transparency and replication of the experiments, the repository provides:

- Screenshots illustrating the agent workflow configuration in the `images_pipelines/` folder.
- The prompts used during the experiments in the `prompts/` folder.

These materials allow readers to understand the experimental setup and replicate the workflow configuration within the AutoGPT platform. 
To run AutoGen locally,  the official repository (which provides the Docker configuration) is available at: 
https://github.com/Significant-Gravitas/AutoGPT

---

#### Dify

The `metagente_optimization.yml` and `metagente_evaluation.yml` files contain the workflows created for the experiments. These workflows can be imported and executed within the Dify platform using the option **Import DSL file**.

To run Dify locally, the official repository (which provides the Docker configuration) is available at:  
https://github.com/langgenius/dify

Once the platform is running, access the Dify interface and import the workflow files (`metagente_optimization.yml` or `metagente_evaluation.yml`) using the **Import DSL file** option.

To execute the `metagente_optimization.yml` workflow, an external API call is required. The implementation of this API is provided in the `dify_API/` folder.

Before running the workflow, start the API service locally after installing the requirements:

```bash
cd dify_API
uvicorn rouge_api:app --host 127.0.0.1 --port 8000 --reload
```
---

#### Semantic Kernel

The `semantic_kernel/METAGENTE` or `semantic_kernel/METAGENTE_agent_chat`  directory contains the implementation based on Semantic Kernel.

Run the experiment:

For optimization workflow
```bash
python main.py
```

For evaluation workflow
```bash
python evaluation.py
```



