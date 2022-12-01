# mabs: A Unified Framework for Multi-Armed Bandit Analysis

A framework for multi-armed bandit agents and environments.

## Running Experiments
```bash
python run.py all  # run all experiments

python run.py etc  # run Explore-Then-Commit (ETC) experiments
python run.py ucb  # run UCB experiments
python run.py ts  # run Thompson Sampling experiments
python run.py optimal  # run Bayesian Optimal Policy experiments

python run.py linucb  # run LinUCB experiments
python run.py lints  # run Linear Thompson Sampling experiments
```

All the results are going to be saved in `results/` sub-directory.

## Visualize Results

Notebooks to visualize collected results could be found in `notebooks/` sub-directory.
