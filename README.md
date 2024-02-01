# Connecting the Dots

![](fig.png)

This repository contains code for creating the results and figures in the ICML 2024 submission *"Connecting the Dots: Is Mode-Connectedness the Key to Feasible Sample-Based Inference in Bayesian Neural Networks?"*.

## Getting Started

Run `pip install -r requirements.txt`. We use python `3.10`.

## File Structure

```
├── data
├── experiments
|   ├── fcn_bnns
|   └── fcn_ensembles
├── notebooks
├── probabilisticml
└── src
```

## Usage

Fit a fully connected Bayesian neural network to a dataset in `data/` with the following steps:

1. Specify the experiment configuration in `experiments/fcn_bnns/conifg.yaml`. The configuration file should be self-explanatory.
2. Run `docker compose -f docker-compose-fcn-bnn.yml up --build` from the root of the repository if you want to monitor the progressbars and prints. Otherwise, run `docker compose -f docker-compose-fcn-bnn.yml up --build -d` to run the experiment in the background.
3. The results will be saved in `experiments/fcn_bnns/results/fcn_bnns/` folder with a timestamped folder name. The results contain the posterior samples, the model configuration `json` file, and the overall config of the experiments and a `txt` file reporting the runtime of the individual experiments.
4. Aggregate some statistics from the results by running `python ../../../experiments/fcn_bnns/aggregate_cross_exp.py` from the experiments folder. This will create a `csv` file with the aggregated statistics in the the same folder. Examples for reported stats are hold out RMSE, LPPD, the configurations of the experiments, and the runtime of the individual experiments.


## Analysis of Results

- explain notebooks
