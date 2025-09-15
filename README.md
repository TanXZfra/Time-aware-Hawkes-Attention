# Hawkes Attention — Official Code

**Official code repository** for the paper 
**"From Hawkes Processes to Attention: Time-Modulated Mechanisms for Event Sequences"**

This repository implements the **Hawkes Attention** model: a time-modulated attention operator for Marked Temporal Point Processes (MTPP) which uses per-type neural kernels $\phi_c(\Delta t)% to directly modulate Q/K/V projections without positional encodings. The code provides training, evaluation, ablation scripts and utilities used to produce the results in the paper.

This project is implemented on top of the **EasyTPP** framework. We strongly recommend reading the EasyTPP paper and repository before running this code, as EasyTPP provides the dataset splits, evaluation code, common utilities, etc., used here.

- EasyTPP (recommended): **[https://github.com/ant-research/EasyTemporalPointProcess]** — _please refer to the official EasyTPP repo for usage and other details._

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```
## Datasets
All datasets should be placed under the dataset/ folder, following the EasyTPP format and preprocessing requirements

```bash
    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl

    configs
     |______experiment_config.yaml
```




---



