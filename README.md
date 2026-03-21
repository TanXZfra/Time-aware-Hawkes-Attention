# Hawkes Attention — Official Code

**Official code repository** for the paper [**"From Hawkes Processes to Attention: Time-Modulated Mechanisms for Event Sequences"**](https://arxiv.org/abs/2601.09220)

This repository implements the **Hawkes Attention** model: a time-modulated attention operator for Marked Temporal Point Processes (MTPP) which uses per-type neural kernels $\phi_c(\Delta t)$ to directly modulate Q/K/V projections without positional encodings. The code provides training, evaluation, ablation scripts and utilities used to produce the results in the paper.

This project is implemented on top of the **EasyTPP** framework. We strongly recommend reading the EasyTPP paper and repository before running this code, as EasyTPP provides the dataset splits, evaluation code, common utilities, etc., used here.

- EasyTPP (recommended): **[https://github.com/ant-research/EasyTemporalPointProcess]** — _please refer to the official EasyTPP repo for usage and other details._

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
## Datasets

All datasets should be placed under the `dataset` folder, following the EasyTPP format and preprocessing requirements. 

More details about the data preprocessing could be found in the EasyTPP repo.

```bash
    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl
```
## Configurations

All training configurations are stored in `.yaml` files under `scripts/train_experiments/`.

Select the dataset config and adjust hyperparameters inside the corresponding YAML file.

```bash
scripts/train_experiments/
  ├── amazon.yaml
  ├── taxi.yaml
  └── ...
```
## Running Experiments

Modify the `run.sh` file to set the correct path to the YAML config file and select the model you want to run.

For example, this file runs an experiment of Hawkes Attention on the taxi dataset:
```bash
python -m scripts.train_experiment.run --config_dir scripts/train_experiment/taxi_config.yaml --experiment_id HawkesTHP_train
```

Then run the `run.sh` file:
```bash
bash run.sh
```



---



