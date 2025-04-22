
# PrO-KGC

This repository is related to our paper:  
**_PrO-KGC: Prompt Optimization for LLM-Based Knowledge Graph Completion_**  
submitted to **LLM-TEXT2KG 2025** — the *4th International Workshop on LLM-Integrated Knowledge Graph Generation from Text (Text2KG)*,  
on **June 01–07, 2025** in **Portorož, Slovenia**

### Abstract

Knowledge Graphs (KGs) are integral to multiple applications which makes their completeness a key research
focus. Large Language Models (LLMs) have been increasingly leveraged alongside embedding models to enhance
Knowledge Graph Completion (KGC). However, effectively transforming contextual features from KGs into well structured textual prompts for LLMs remains a challenge. In this work, we propose PrO-KGC, a framework for
Prompt Optimization in the context of LLM-Based KGC. PrO-KGC aims to enhance LLM-based KGC performance
by refining and optimizing input prompts through multiple steps. This process includes enriching prompts
with additional information, incorporating structural composition patterns and facts, and refining relation
representations to be more LLM-friendly. Experimental results demonstrate the effectiveness of our approach,
achieving improvements across three common benchmarks compared to vanilla models.


### For reproducibility:
Make sure you have Python installed (we used the 3.12.1 version), then:
* Install requirements through:
`pip install -r requirements.txt`

* Download datasets:
   * Codex-m from: https://github.com/tsafavi/codex/tree/master

   * FB15k-237 and WN18RR from: https://github.com/yao8839836/kg-bert/tree/master/data



* Setup your OpenAI API key: Create a `.env` file in the project root and add the following line:
   ```env 
   OPENAI_API_KEY=your_openai_api_key_here
   ```