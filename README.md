# The Relationship Between Reasoning and Performance in Large Language Models--o3 mini Thinks Harder, Not Longer

This repository contains the analysis code for "The Relationship Between Reasoning and Performance in Large Language Models--o3 (mini) Thinks Harder, Not Longer" by Marthe Ballon, Andres Algaba and Vincent Ginis (https://arxiv.org/abs/2502.15631).

## Data
1. Original Omni-MATH dataset by Gao et al. (2024) is available under https://github.com/KbsdJames/Omni-MATH.
2. Replication data for our analysis is available under https://zenodo.org/records/14878936.

Download the data files at the links provided above and insert into the data folder.

## Models 
We evaluate the performance and token use of OpenAI models gpt-4o-06-08-2024, o1-mini-12-09-2024, o3-mini-31-01-2025 medium (default) and o3-mini-31-01-2025 high on the Omni-MATH benchmark. The o3-mini high model, instead of medium, is obtained by setting reasoning_effort to high.

To correct the answers of the considered models, we employ the open-source large language model Omni-Judge by Gao et al. (2024), available at https://huggingface.co/KbsdJames/Omni-Judge. 

## Overview of the code
The performance_eval.py code is based on the evaluation code of the Omni-MATH paper by Gao et al (2024), available at https://github.com/KbsdJames/Omni-MATH.

- config.py: prompts to generate model completions, judge their answers on Omni-MATH
- figures_appendix.ipynb: notebook to reproduce supplementary figures in our manuscript
- figures_main.ipynb: notebook to reproduce main figures in our manuscript
- performance_eval.py: auxiliary functions, performance/token evaluation functions necessary for the analysis
- regression_analysis.ipynb: notebook to reproduce the regression analysis in our manuscript
