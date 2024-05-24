from gnninterpreter import *
import random
from collections import defaultdict, namedtuple
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from tqdm.auto import tqdm, trange

import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score

import os

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # Get 5 random seeds
    seeds = list(range(5))

    # Create 'logs' directory (if it doesn't exist)
    logs_directory = "logs"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    motiflog(seeds)

def motiflog(seeds):
    with open("logs/train_times_motif.txt", "w") as log_file_train, open("logs/class_probs_motif.txt", "w") as log_file_probs:
        # Write header
        log_file_train.write("Seed\tClass\tTrain Time\n")
        log_file_probs.write("Seed\tClass\tMean\tStd\n")

        for _, seed in enumerate(seeds):
            # Initialize dataset
            motif = MotifDataset(seed=seed)

            # Seed all
            seed_all(seed)

            # Load model
            model = GNNClassifier(hidden_channels=64,
                      node_features=len(motif.NODE_CLS),
                      num_classes=len(motif.GRAPH_CLS),
                      num_layers=3)
            model.load_state_dict(torch.load('models_checkpoints/motif_gnn_64x3.pt'))

            # Generate average embedding
            embeds = [[] for _ in range(len(motif.GRAPH_CLS))]
            with torch.no_grad():
                for data in tqdm(motif):
                    embeds[data.y.item()].append(model.eval()(pyg.data.Batch.from_data_list([data]))["embeds"].numpy())
            mean_embeds = [torch.tensor(np.concatenate(e).mean(axis=0)) for e in embeds]

            trainer = {}
            sampler = {}

            # Class 1
            house(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs)
            seed_all(seed)
            house_x(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs)
            seed_all(seed)
            comp_4(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs)
            seed_all(seed)
            comp_5(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs)



    return
        
def house(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs):
    classes = 1
    sampler[classes] = s = GraphSampler(max_nodes=20,
                                        num_node_cls=len(motif.NODE_CLS),
                                        temperature=0.15,
                                        learn_node_feat=True)

    criterion = nn.Sequential(
        WeightedCriterion([
            # dict(key="logits", criterion=ClassScoreCriterion(class_idx=0), mode="maximize", weight=0.5),
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=1), mode="maximize", weight=1),
            # dict(key="logits", criterion=ClassScoreCriterion(class_idx=2), mode="maximize", weight=-0.5),
            # dict(key="logits", criterion=ClassScoreCriterion(class_idx=3), mode="maximize", weight=-0.5),
            # dict(key="logits", criterion=CrossEntropyBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=MeanSquareBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=DynamicBalancingCriterion(classes=classes), weight=5),
            # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes[0]]), weight=10),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[1]), weight=1),
        ]),
        NormPenalty(lambda: s.omega, order=1, weight=1), # L1 penalty on omega, encourage uncertainty
        NormPenalty(lambda: s.omega, order=2, weight=1), # L2 penalty on omega, avoid extreme probabilities
        NormPenalty(lambda: s.xi, order=1, weight=0), # L1 penalty on xi
        NormPenalty(lambda: s.xi, order=2, weight=0), # L2 penalty on xi
        budget := BudgetPenalty(lambda: s.theta, budget=10, order=2, beta=0.5, weight=1), # Budget penalty on theta, encourage sparsity
        CrossEntropyPenalty(lambda: s.theta, weight=0), # Element-wise entropy penalty on theta, encourage discreteness
        KLDivergencePenalty(lambda: tuple(s.theta_pairs), binary=True, weight=0), # Pair-wise cross entropy on E, encourage connectivity
    )
    optimizer = torch.optim.SGD(s.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # construct graph sampler trainer
    trainer[classes] = Trainer(sampler=s,
                    discriminator=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dataset=motif,
                    k_samples=1,
                    seed=seed,
                    classes=classes)

    def penalty_cond(out, trainer, boundary_cls=classes):
        return out["probs"][0, classes].item() > 0.9
    def break_cond(out, trainer, boundary_cls=classes):
        return all([
            penalty_cond(out, trainer, boundary_cls),
            trainer.sampler.expected_m < 15
        ])

    trainer[1].train(2000, dynamic_penalty=budget, penalty_cond=penalty_cond, break_cond=break_cond, log_file=log_file_train)
    trainer[1].evaluateAndLog(threshold=0.5, log_file_probs=log_file_probs)

def house_x(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs):
    classes = 2
    sampler[classes] = s = GraphSampler(max_nodes=20,
                                        num_node_cls=len(motif.NODE_CLS),
                                        temperature=0.15,
                                        learn_node_feat=True)

    criterion = nn.Sequential(
        WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=classes), mode="maximize", weight=1),
            # dict(key="logits", criterion=CrossEntropyBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=MeanSquareBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=DynamicBalancingCriterion(classes=[classes]), weight=1),
            # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes[0]]), weight=10),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes]), weight=1),
        ]),
        NormPenalty(lambda: s.omega, order=1, weight=1),  # L1 penalty on omega, encourage uncertainty
        NormPenalty(lambda: s.omega, order=2, weight=1),  # L2 penalty on omega, avoid extreme probabilities
        NormPenalty(lambda: s.xi, order=1, weight=0),  # L1 penalty on xi
        NormPenalty(lambda: s.xi, order=2, weight=0),  # L2 penalty on xi
        budget := BudgetPenalty(lambda: s.theta, budget=10, order=2, beta=0.5, weight=1),
        # Budget penalty on theta, encourage sparsity
        CrossEntropyPenalty(lambda: s.theta, weight=0),  # Element-wise entropy penalty on theta, encourage discreteness
        KLDivergencePenalty(lambda: tuple(s.theta_pairs), binary=True, weight=0),
        # Pair-wise cross entropy on E, encourage connectivity
    )
    optimizer = torch.optim.SGD(s.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # construct graph sampler trainer
    trainer[classes] = Trainer(sampler=s,
                               discriminator=model,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               dataset=motif,
                               k_samples=16,
                               seed=seed,
                               classes=classes)

    def penalty_cond(out, trainer, boundary_cls=classes):
        return out["probs"][0, classes].item() > 0.9

    def break_cond(out, trainer, boundary_cls=classes):
        return all([
            penalty_cond(out, trainer, boundary_cls),
            trainer.sampler.expected_m < 15
        ])

    trainer[2].train(2000, dynamic_penalty=budget, penalty_cond=penalty_cond, break_cond=break_cond, log_file=log_file_train)
    trainer[2].evaluateAndLog(threshold=0.5, log_file_probs=log_file_probs)
def comp_4(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs):
    classes = 3
    sampler[classes] = s = GraphSampler(max_nodes=20,
                                        num_node_cls=len(motif.NODE_CLS),
                                        temperature=0.15,
                                        learn_node_feat=True)

    criterion = nn.Sequential(
        WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=classes), mode="maximize", weight=1),
            # dict(key="logits", criterion=CrossEntropyBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=MeanSquareBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=DynamicBalancingCriterion(classes=[classes]), weight=1),
            # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes[0]]), weight=10),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes]), weight=1),
        ]),
        NormPenalty(lambda: s.omega, order=1, weight=1),  # L1 penalty on omega, encourage uncertainty
        NormPenalty(lambda: s.omega, order=2, weight=1),  # L2 penalty on omega, avoid extreme probabilities
        NormPenalty(lambda: s.xi, order=1, weight=0),  # L1 penalty on xi
        NormPenalty(lambda: s.xi, order=2, weight=0),  # L2 penalty on xi
        budget := BudgetPenalty(lambda: s.theta, budget=10, order=2, beta=0.5, weight=1),
        # Budget penalty on theta, encourage sparsity
        CrossEntropyPenalty(lambda: s.theta, weight=0),  # Element-wise entropy penalty on theta, encourage discreteness
        KLDivergencePenalty(lambda: tuple(s.theta_pairs), binary=True, weight=0),
        # Pair-wise cross entropy on E, encourage connectivity
    )
    optimizer = torch.optim.SGD(s.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # construct graph sampler trainer
    trainer[classes] = Trainer(sampler=s,
                               discriminator=model,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               dataset=motif,
                               k_samples=16,
                               seed=seed,
                               classes=classes)

    def penalty_cond(out, trainer, boundary_cls=classes):
        return out["probs"][0, classes].item() > 0.9

    def break_cond(out, trainer, boundary_cls=classes):
        return all([
            penalty_cond(out, trainer, boundary_cls),
            trainer.sampler.expected_m < 15
        ])

    trainer[3].train(2000, dynamic_penalty=budget, penalty_cond=penalty_cond, break_cond=break_cond, log_file=log_file_train)
    trainer[3].evaluateAndLog(threshold=0.5, log_file_probs=log_file_probs)
def comp_5(motif, mean_embeds, model, trainer, sampler, seed, log_file_train, log_file_probs):
    classes = 4
    sampler[classes] = s = GraphSampler(max_nodes=20,
                                        num_node_cls=len(motif.NODE_CLS),
                                        temperature=0.15,
                                        learn_node_feat=True)

    criterion = nn.Sequential(
        WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=classes), mode="maximize", weight=1),
            # dict(key="logits", criterion=CrossEntropyBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=MeanSquareBoundaryCriterion(class_a=0, class_b=1), weight=1),
            # dict(key="logits", criterion=DynamicBalancingCriterion(classes=[classes]), weight=1),
            # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes[0]]), weight=10),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[classes]), weight=1),
        ]),
        NormPenalty(lambda: s.omega, order=1, weight=1),  # L1 penalty on omega, encourage uncertainty        NormPenalty(lambda: s.omega, order=2, weight=1),  # L2 penalty on omega, avoid extreme probabilities
        NormPenalty(lambda: s.xi, order=1, weight=0),  # L1 penalty on xi
        NormPenalty(lambda: s.xi, order=2, weight=0),  # L2 penalty on xi
        budget := BudgetPenalty(lambda: s.theta, budget=10, order=2, beta=0.5, weight=1),
        # Budget penalty on theta, encourage sparsity
        CrossEntropyPenalty(lambda: s.theta, weight=0),  # Element-wise entropy penalty on theta, encourage discreteness
        KLDivergencePenalty(lambda: tuple(s.theta_pairs), binary=True, weight=0),
        # Pair-wise cross entropy on E, encourage connectivity
    )
    optimizer = torch.optim.SGD(s.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # construct graph sampler trainer
    trainer[classes] = Trainer(sampler=s,
                               discriminator=model,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               dataset=motif,
                               k_samples=16,
                               seed=seed,
                               classes=classes)

    # %%
    def penalty_cond(out, trainer, boundary_cls=classes):
        return out["probs"][0, classes].item() > 0.9

    def break_cond(out, trainer, boundary_cls=classes):
        return all([
            penalty_cond(out, trainer, boundary_cls),
            trainer.sampler.expected_m < 15
        ])

    trainer[4].train(2000, dynamic_penalty=budget, penalty_cond=penalty_cond, break_cond=break_cond, log_file=log_file_train)
    trainer[4].evaluateAndLog(threshold=0.5, log_file_probs=log_file_probs)

if __name__ == '__main__':
    main()