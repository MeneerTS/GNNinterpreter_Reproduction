import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from XGNN_stripped_repo.XGNN import gnn_explain

import time
import torch
import random
import numpy as np

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Get 5 random seeds
    seeds = list(range(5))

    # Create 'logs' directory (if it doesn't exist)
    logs_directory = "logs"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    xgnnlog(seeds)

def xgnnlog(seeds):
    with open("logs/train_times_xgnn.txt", "w") as log_file_train, open("logs/class_probs_xgnn.txt", "w") as log_file_probs:
        # Write header
        log_file_train.write("Seed\tClass\tIterations\tTrain Time\n")
        log_file_probs.write("Seed\tClass\tProbabilities\n")

        for _, seed in enumerate(seeds):
            seed_all(seed)

            # Figure 4 (start from 1 molecule of C)
            max_nodes = [5, 6, 7]
            for _, val in enumerate(max_nodes):
                # Class 1 (mutagenic)
                explainer = gnn_explain.gnn_explain(max_node=val, max_step=30, target_class=1, max_iters=100)
                explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=1, seed=seed, startmol=0, maxnodes=val)

            for _, val in enumerate(max_nodes):
                # Class 0 (non-mutagenic)
                explainer = gnn_explain.gnn_explain(max_node=val, max_step=30, target_class=0, max_iters=100)
                explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=0, seed=seed, startmol=0, maxnodes=val)
            
            # Figure 5 (start from one molecule of C, N and O with max 5 nodes)
            labels = [0, 1, 2] # initial molecule to start from
            for _, val in enumerate(labels):
                # Class 0 (nonmutagenic)
                explainer = gnn_explain(max_node=5, max_step=30, target_class=0, max_iters=100)
                explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=0, seed=seed, startmol=val, maxnodes=5)

                # Class 1 (mutagenic)
                explainer = gnn_explain(max_node=5, max_step=30, target_class=1, max_iters=100)
                explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=1, seed=seed, startmol=val, maxnodes=5)

            # Extra test for train time with different number of iterations
            # iterations = [200, 300, 400, 500]
            # for _, iteration in enumerate(iterations):
            #     max_nodes = [6, 7]
            #     for _, val in enumerate(max_nodes):
            #         # Class 1 (mutagenic)
            #         explainer = gnn_explain(max_node=val, max_step=30, target_class=1, max_iters=iteration)
            #         explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=1, seed=seed, startmol=0, maxnodes=val, onlyLogTrainTime=True)

            #     for _, val in enumerate(max_nodes):
            #         # Class 0 (non-mutagenic)
            #         explainer = gnn_explain(max_node=val, max_step=30, target_class=0, max_iters=iteration)
            #         explainer.train(log_file_train=log_file_train, log_file_probs=log_file_probs, classes=0, seed=seed, startmol=0, maxnodes=val, onlyLogTrainTime=True)
            

if __name__ == '__main__':
    main()

    