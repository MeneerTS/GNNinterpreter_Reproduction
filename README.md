# GNNInterpreter Reproducibility Study

#### Ana-Maria Vasilcoiu, Batu Helvacioğlu, Thies Kersten, Thijs Stessens

---

This repo contains the official implementation for the reproducibility study conducted based on the paper entitled "[GNNInterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks](https://openreview.net/forum?id=rqq6Dh8t4d)".

## Structure of the repository
The repository is organized in 2 main folders, namely:
- `GNNInterpreter-Initial-Version` = contains the earlierst version of the authors original repository (https://github.com/yolandalalala/GNNInterpreter/commit/a419343d0de20674e14cd1051b7983981cf6b47c)
- `GNNInterpreter-Most-Recent-Version` = contains the latest updated on the authors original repository (https://github.com/yolandalalala/GNNInterpreter/tree/main)

Inside GNN-Interpreter-Initial-Version, a stripped version of the XGNN original repository can be found, in the XGNN_stripped_repo folder ([XGNN paper](https://arxiv.org/abs/2006.02587), [XGNN repository](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN)).

Moreover, both directories contain the following: (1) a `gnninterpreter` folder with all the model specific code files and (2) some test files based on which the reproduction experiments were conducted. 

## Reproducibility test files

The reproduction files in the initial version are the following:
- `cyclicity_explanation.py`, `motif_explanation.py`, `mutag_explanation.py`, `shape_explanation.py`, `xgnn_explanation.py` - which can all be run separately to conduct experiments on one intended dataset / model (exception XGNN)
- `datasets_logging.py` - which runs experiments for GNNInterpreter on all 4 datasets and for XGNN

If you run either of these files, the logging information is going to be saved in a newly created folder called `logs`, while the explanation graphs are going to be saved in a separate folder for each experiments, for example under the name `CyclicityDatasetplots`.

On the new version of the repository we only tested Cyclicity dataset. The test file which can be ran to verify this is `cyclicity_logging.py`.

We have included in the repo the final results of our experiments. For the Cyclicity dataset you can find these in `GNNInterpreter-Most-Recent-Version\logs` and `GNNInterpreter-Most-Recent-Version\CyclicityDatasetPlots`. For all the other datasets, you can find them in `GNNInterpreter-Initial-Version\reproduction_logs` and `GNNInterpreter-Initial-Version\reproduction_explanations`. 

Both versions also contain a `log_reader.py` file that reads the logging files and outputs some readble statistics about each class in the respective dataset. If you run this file, make sure to modify the `accuracy_paths` and `training_paths` variables in the main method of this file to point to the correct files you want to read (e.g. if you want to check the reproduction results, change the path to `reproduction_logs`).

## Original paper extensions

We have extended the original experiments with testing GNNInterpreter on a different real-world dataset, namely the Reddit-Binary dataset. The GNN model for this dataset has been trained and saved in a checkpoint in the newest version, thus the respective files can be found inside `GNNInterpreter-Most-Recent-Version`, namely `reddit_model_training.py`, `reddit_interpreter.py` and `ckpts/reddit.pt`. 

## Environment and Installation

All reproducibility experiments have been conducted in an environment which can be installed using `fact_environment.yml`. We are using Python 3.9.0, PyTorch 2.0.0 and PyTorch Geometric 2.3.0. To fully replicate our environment, you need to separately install 2 libraries using pip, as follows:
```
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
## Changes from original Github

Some modifactions and additions to the original repository were made. These include:

1. Adding the calculate accuracy method to base_graph_dataset.py.
2. Added seed and class to the variables of the Trainer class, for logging purposes.
3. Removing the line of code that called torch.seed() in the sampling method of graph_sampler.py. This was done to prevent true randomness from creating inconsistencies.
4. Adding the method evaluateAndLog() in Trainer.py.
5. Adjusted train() in train.py to accommodate logging.

