from .gnninterpreter import *
import torch
import random
import numpy as np
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

    cyclicitylog(seeds,explain_all_classes=True)

def cyclicitylog(seeds, train_times_log_path="logs/train_times_cyclicity_new.txt", class_probs_log_path="logs/class_probs_cyclicity_new.txt",
             pretrained_model_checkpoint_path='ckpts/cyclicity.pt',
             explain_all_classes=False,explain_red_cyclic=False,explain_green_cyclic=False,explain_acyclic=False):
    with open(train_times_log_path, "w") as log_file_train, open( class_probs_log_path, "w") as log_file_probs:
        # Write header
        log_file_train.write("Seed\tClass\tTrain Time\n")
        log_file_probs.write("Seed\tClass\tMean\tStd\n")

        for _, seed in enumerate(seeds):
            # Initialize dataset
            dataset = CyclicityDataset(seed=seed)

            # Seed all
            seed_all(seed)

            # Model
            model = NNConvClassifier(node_features=len(dataset.NODE_CLS),
                                edge_features=len(dataset.EDGE_CLS),
                                num_classes=len(dataset.GRAPH_CLS),
                                hidden_channels=32)
            model.load_state_dict(torch.load(pretrained_model_checkpoint_path))

            # Mean embeddings
            mean_embeds = dataset.mean_embeddings(model)

            # Initialize trainer
            trainer = classTrainer(dataset, model, mean_embeds, seed)

            if explain_all_classes:
                explain_red_cyclic = True
                explain_green_cyclic = True
                explain_acyclic = True
            elif not (explain_red_cyclic or explain_green_cyclic or explain_acyclic):
                print("No class selected for training, please specify a class or train_all_classes as True.")

            # Red cyclic
            if explain_all_classes:
                trainer[0].train(2000, log_file=log_file_train)
                trainer[0].evaluateAndLog(log_file_probs=log_file_probs)

            # Green cyclic
            if explain_green_cyclic:
                seed_all(seed)
                trainer[1].train(2000, log_file=log_file_train)
                trainer[1].evaluateAndLog(log_file_probs=log_file_probs)

            # Acylic
            if explain_acyclic:
                seed_all(seed)
                trainer[2].train(2000, log_file=log_file_train)
                trainer[2].evaluateAndLog(log_file_probs=log_file_probs)

    return
        
def classTrainer(dataset, model, mean_embeds, seed):
    trainer = {}
    sampler = {}

    cls_idx = 0
    trainer[cls_idx] = Trainer(
        sampler=(s := GraphSampler(
            max_nodes=20,
            num_edge_cls=len(dataset.EDGE_CLS),
            temperature=0.15,
            learn_node_feat=False,
            learn_edge_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),
            dict(key="logits", criterion=MeanPenalty(), weight=1),
            dict(key="omega", criterion=NormPenalty(order=1), weight=2),
            dict(key="omega", criterion=NormPenalty(order=2), weight=2),
            # dict(key="xi", criterion=NormPenalty(order=1), weight=0),
            # dict(key="xi", criterion=NormPenalty(order=2), weight=0),
            dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=5),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=15, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10,
        seed=seed,
        classes=cls_idx
    )

    cls_idx = 1
    trainer[cls_idx] = Trainer(
        sampler=(s := GraphSampler(
            max_nodes=20,
            num_edge_cls=len(dataset.EDGE_CLS),
            temperature=0.15,
            learn_node_feat=False,
            learn_edge_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),
            dict(key="logits", criterion=MeanPenalty(), weight=1),
            dict(key="omega", criterion=NormPenalty(order=1), weight=2),
            dict(key="omega", criterion=NormPenalty(order=2), weight=2),
            # dict(key="xi", criterion=NormPenalty(order=1), weight=0),
            # dict(key="xi", criterion=NormPenalty(order=2), weight=0),
            dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=5),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=15, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10,
        seed=seed,
        classes=cls_idx
    )

    cls_idx = 2
    trainer[cls_idx] = Trainer(
        sampler=(s := GraphSampler(
            max_nodes=20,
            num_edge_cls=len(dataset.EDGE_CLS),
            temperature=0.15,
            learn_node_feat=False,
            learn_edge_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),
            dict(key="logits", criterion=MeanPenalty(), weight=1),
            dict(key="omega", criterion=NormPenalty(order=1), weight=2),
            dict(key="omega", criterion=NormPenalty(order=2), weight=2),
            # dict(key="xi", criterion=NormPenalty(order=1), weight=0),
            # dict(key="xi", criterion=NormPenalty(order=2), weight=0),
            dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=5),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=15, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10,
        seed=seed,
        classes=cls_idx
    )

    return trainer


if __name__ == '__main__':
    main()
    