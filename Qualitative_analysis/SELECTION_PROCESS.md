# Selection process for functionalities and frameworks 

This document describes the methodology used to identify, filter, and analyze gray literature following the  **multi-vocal study approach** and established guidelines for systematic and tertiary studies in software engineering. The study considers both white and gray literature. In addition, we select the repositories based on Github popularity metrics. 


The methodology is structured into three main phases, 1) **Planning**, 2) **Conducting**, and 3) **Mapping**. 



---

## 1. Planning Phase

### Data Source

We selected **Scopus** as the primary digital library due to, broad coverage of peer-reviewed literature. 

---

### Search Strategy

The search query is composed of two keyword group. The final query is reported below. 



(mas OR multi-agent AND system AND llm OR large-language AND models
OR pre-trained AND language AND model
AND "evidence-based software engineering" OR survey OR "structured review"
OR "systematic review" OR "literature review" OR "literature analysis"
OR "in-depth survey" OR "literature survey" OR "meta-analysis"
OR "past studies" OR "subject matter expert"
OR "analysis of research" OR "empirical body of knowledge")




### Inclusion Criteria (Literature)

- Surveys or systematic studies on **MAS + LLMs**
- Papers describing **foundational MAS concepts**
- Publications in **high-ranked venues**
- English language
- Includes **preprints** (e.g., arXiv)

---

###  Exclusion Criteria (Literature)

- MAS papers without LLM focus
- Papers proposing new MAS without analysis
- Workshops, posters, short/vision papers

---

## MAS Framework Selection (GitHub)

### Sources

The primary source was the GitHub awesome list available [here](https://github.com/kaushikb11/awesome-llm-agents). In addition we search for blogs, newsletter, and gray literature to find more. 


---

###  Inclusion Criteria (Frameworks)

- Open-source with **GitHub repository**
- Proper **documentation/tutorials**
- Permissive license

---

### Exclusion Criteria (Frameworks)

- Single-agent or non-orchestrated systems
- Less than **10,000 GitHub stars**
- Paid/subscription-only tools

As filter criteria, we emplöoyed traditional GitHub popularity metrics. e.g., stars,forks, contributors. However, we select also frameworks that offers a good coverage of the indetified functionalities (F1-F10) to achive better coverage.  

---

##  2. Conducting Phase

### Literature Filtering

We start from 47 papers, ending up into 18 after the application of the criteria. Although we acknowledge that the field is fast-evolving, we created a "quasi-gold" set to indetify pivotal functionalities. The full list of the papers, including the rationale for inclusion/exclusion, is reported in Survey_all_papers.csv file [here](https://github.com/MDEGroup/LLMs-based-MAS-ReplicationPackage/blob/main/Qualitative_analysis/Survey_all_papers.csv). 




### Framework Filtering

1. Initial pool: **20 frameworks**
2. Excluded:
   - 2 frameworks (<10k stars)
   - 2 frameworks (subscription required)
3. Final set: **16 frameworks**
4. Analyzed frameworks in the qualitative analysis: **8 frameworks** 

---

## 3. Mapping Phase

To extract the functionalities, two co-authors Read full text of selected papers, extract initial MAS-related concepts, and merge the results via discussion. From an initial set of **57** functionalities, we ended up into **10**. 

---

## MAS Functionalities

| ID  | Feature | Description |
|-----|--------|------------|
| **F1** | MAS Core Architecture | Support for agents, orchestration, memory, and tools |
| **F2** | MAS Type | Homogeneous vs. Heterogeneous agents |
| **F3** | Role Specification | Ability to define agent roles |
| **F4** | Tool Support | Integration with APIs/external tools |
| **F5** | Remote Access | Web or remote interaction capabilities |
| **F6** | Agent Monitoring | Metrics, telemetry, evaluation |
| **F7** | Human Feedback | Human-in-the-loop support |
| **F8** | Agent Comparison | Compare agents via UI/API |
| **F9** | Benchmark Reuse | Support for datasets and evaluation |
| **F10** | Discovery Capabilities | Agent/tool marketplaces or discovery |

---


Then, we define seven characteristc to evaluate qualitatively the frameworks as follows: 


## Characteristics of the selected frameworks

| Alias | Characteristic | Description |
|------|--------------|------------|
| **C1** | Installation | Describes how to install the tool, including dependencies and prerequisites. Includes instructions for local setup and necessary configurations. |
| **C2** | Developer Interface | Evaluates ease of use and intuitiveness of the interface. Includes quality of documentation, clarity, completeness, tutorials, examples, and API references. |
| **C3** | Model and Tools Integration | Support for integrating different AI models (e.g., LLMs) and external services. Assesses flexibility, availability of pre-built integrations, and support for custom integrations. |
| **C4** | Agent Creation | Support for creating and managing agents, including templates, examples, role definition, and integration of external knowledge. |
| **C5** | Agent Orchestration | Support for orchestrating agents and managing interactions, including workflow definition and coordination patterns. |
| **C6** | Monitoring | Support for monitoring and debugging agents, including logging, telemetry, performance tracking, and issue identification. |


