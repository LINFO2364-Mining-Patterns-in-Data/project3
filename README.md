# Bayesian Networks Project

## Project Overview
This project focuses on Bayesian Network structure learning, parameter estimation, and inference. It includes tools for learning Bayesian Network structures from data, performing inference, and evaluating models.

## Requirements
- Required libraries: `pandas`, `numpy`


## How to Run

### Step 1: Posterior Query

Run a single posterior query on a .bif network:

```
python step1.py <network.bif> '{"Var1":"value1", "Var2":"value2"}'
```

examples : 
```
python step1.py datasets/sprinkler/sprinkler_complete.bif '{"Cloudy":"1","Sprinkler":"0"}'
python step1.py datasets/sprinkler/sprinkler_complete.bif '{"Sprinkler":"1","Cloudy":"1","Rain":"0"}'    
python step1.py datasets/sprinkler/sprinkler_complete.bif '{"Sprinkler":"1"}' 
```


### Step 2: Learn Parameters

Learn parameters for a given structure:

```
python step2.py <structure.bif> <train.csv> <out_complete.bif>
```

example : 
```
python3 step2.py datasets/sprinkler/sprinkler_complete.bif datasets/sprinkler/train.csv datasets/sprinkler/sprinkler_learned_complete.bif
```

### Step 3: Learn Structure
Learn structure and parameters from scratch:

```
python step3.py <train.csv> <out_learned.bif>
```
example : 
```
python3 step3.py datasets/sprinkler/train.csv datasets/sprinkler/out.bif
```

### Step 4: Evaluate and Save Best Network
Evaluate the provided structure and learn a new one:

```
python step4.py <train.csv> <test.csv> <test_missing.csv> <out.bif>
```
example : 
```
python step4.py datasets/sprinkler/train.csv datasets/sprinkler/test.csv datasets/sprinkler/test_missing.csv datasets/sprinkler/outnamefile.bif
```

### Run Experiments

Running the experiments will generate a CSV file located at `results/experiment_results.csv`. This file contains detailed data for reporting, including:

- Dataset name
- Timestamp of the experiment
- Accuracy metrics for different configurations
- Learned structure details (edges, max in-degree)
- Execution time

For that, 

1. Edit run_exp.py to specify datasets.
2. Run:
```
python run_exp.py
```
