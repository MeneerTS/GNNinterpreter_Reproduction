from gnninterpreter import *
import torch
from torch import nn
import random
import numpy as np
import time
from tqdm.auto import trange
import networkx as nx
import matplotlib.pyplot as plt

#This file contains methods used for finetuning and for generating the results for the dataset
#In the main method you can uncomment the method you want to use


def accuracy_f1():
    #Accuracy of the pretrained model
    reddit = Redditdataset(seed=12345) #Same seed as used to train it according to notebook, so should give them same split
    reddit_train, reddit_val = reddit.train_test_split(k=10)
    model = GCNClassifier(node_features=len(reddit_train.NODE_CLS),
                        num_classes=len(reddit_train.GRAPH_CLS),
                        hidden_channels=64,
                        num_layers=5)
    model.load_state_dict(torch.load('ckpts/reddit.pt'))
    print("Stats of pretrained model on test + train set:")
    print("Val Accuracy: {}".format(reddit_val.calculate_accuracy(model)))
    print("Val F1 scores: {}".format(reddit_val.evaluate_model(model)))

def test_interpreter(show=False):
    dataset = Redditdataset(seed=12345)
    model = GCNClassifier(node_features=1,
                        num_classes=2,
                        hidden_channels=64,
                        num_layers=5)
    mean_embeds = dataset.mean_embeddings(model)
    for seed in [12,13,14,15]: #seed 26, 69, 123 and 42  and 420 
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        
        model.load_state_dict(torch.load('ckpts/reddit.pt'))

        print("="*30)
        print("="*30)
        print("Seed: {}".format(seed))
        print("="*30)     
        print("="*30)

        
        trainer = get_trainer(dataset,model,mean_embeds,seed)
               
        print("class 0")
        start_time = time.time()
        trainer[0].train(1000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time to train: {}".format(elapsed_time))
        print(trainer[0].quantatitive())
        if show:
            for i in range(1):
                trainer[0].evaluate(threshold=0.5, show=True,bernoulli=True,connected=True)  
        print("="*30)

        #class 1
        
        print("class 1")
        start_time = time.time()
        trainer[1].train(1000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time to train: {}".format(elapsed_time))
        print(trainer[1].quantatitive())
        if show:            
            trainer[1].evaluate(threshold=0.5, show=True,bernoulli=True,connected=True)  
        
        #trainer[0].warmup(500,0,1)
        
        
        print("="*30)
def info_dataset():
    reddit = Redditdataset(seed=12345)
    
    
    model = GCNClassifier(node_features=1,
                        num_classes=2,
                        hidden_channels=64,
                        num_layers=5)
    mean_embeds = reddit.mean_embeddings(model)
    trainer = get_trainer(reddit,model,mean_embeds,12345)
    print("Random baseline, class 0: {}".format(trainer[0].quantatitive_baseline()))
    print("Random baseline, class 1: {}".format(trainer[1].quantatitive_baseline()))

   

    
    graph_classes = reddit.y.unique()

    
    class_counts = {class_label.item(): 0 for class_label in graph_classes}
    class_sum_nodes = {class_label.item(): 0 for class_label in graph_classes}
    class_sum_edges = {class_label.item(): 0 for class_label in graph_classes}

    #
    for data in reddit:
        class_label = data.y.item()
        
        
        class_counts[class_label] += 1
        
       
        class_sum_nodes[class_label] += data.num_nodes
        class_sum_edges[class_label] += data.num_edges

    
    for class_label in graph_classes:
        class_label = class_label.item()
        count = class_counts[class_label]
        avg_nodes = class_sum_nodes[class_label] / count
        avg_edges = class_sum_edges[class_label] / count

        print(f"Class {class_label}: {count} graphs, Average Nodes: {avg_nodes:.2f}, Average Edges: {avg_edges:.2f}")

def test_GCN():
    for seed in [123, 26, 69, 42,420]:
        print("="*30)
        print(seed)
        print("="*30)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        reddit = Redditdataset(seed=seed)        
        reddit_train, reddit_val = reddit.train_test_split(k=10)
        reddit_model = GCNClassifier(node_features=1,
                                num_classes=2,
                            hidden_channels=64,
                            num_layers=5)
        for epoch in trange(128):
            train_loss = reddit_train.fit_model(reddit_model, lr=0.01)
            train_f1 = reddit_train.evaluate_model(reddit_model)
            val_f1 = reddit_val.evaluate_model(reddit_model)
        print(f'Epoch: {epoch:03d}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Train F1: {train_f1}, '
                f'Test F1: {val_f1}')
        print("Accuracy: {}".format(reddit_val.calculate_accuracy(reddit_model)))
        
        
    
def get_trainer(dataset,model,mean_embeds,seed):
    # Create trainer class 1
    trainer = {}
    sampler = {}
    cls_idx = 1
    trainer[cls_idx] = Trainer(
        seed=seed,
        classes=[0,1],
        sampler=(s := GraphSampler(
            max_nodes=50,
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,                                                #Tau, temperature
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1), #mu, cosine sim
            dict(key="logits", criterion=MeanPenalty(), weight=0),      #not mentioned anywhere?      
            dict(key="omega", criterion=NormPenalty(order=1), weight=10),   #L1
            dict(key="omega", criterion=NormPenalty(order=2), weight=5),    #L2
            #no node or edge data
            #dict(key="xi", criterion=NormPenalty(order=1), weight=10),      #L1
            #dict(key="xi", criterion=NormPenalty(order=2), weight=5),       #L2
            # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=20),  #Rc, see formula 9
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=4000, order=2, beta=1),           #Rb
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10                                                        #K                                                                  
    )

    

    #Trainer class 0
    cls_idx = 0
    trainer[cls_idx] = Trainer(
        seed=seed,
        classes=[0,1],
        sampler=(s := GraphSampler(
            max_nodes=50,                   #30
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),#1 logits from the GNN
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),#1 similarity to original dataset
            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=5), #5
            dict(key="omega", criterion=NormPenalty(order=2), weight=2), #2
            #dict(key="xi", criterion=NormPenalty(order=1), weight=5),
            #dict(key="xi", criterion=NormPenalty(order=2), weight=2),
            # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=18),#18 encourage connectivity
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=530, order=2, beta=1),  #140 encourage sparsity
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10
    )
    return trainer

def view_data():
    
    reddit = Redditdataset(seed=12345)
    for i in range(200,300):
        data = reddit[i]
        
        edge_index = data.edge_index
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())

        
        node_labels = data.y.tolist()
        if True or data.y.item() == 0:
            
            print(data.y.item())
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos,
                                    
                                    nodelist=G.nodes,
                                    node_size=100,
                                    edgecolors='black')
                
            nx.draw_networkx_edges(G.subgraph(G.nodes), pos, width=1,edge_color='tab:gray')

            plt.show()
    
    

    plt.show()
def get_baseline():
    dataset = Redditdataset(seed=12345)
    model = GCNClassifier(node_features=1,
                        num_classes=2,
                        hidden_channels=64,
                        num_layers=5)
    mean_embeds = dataset.mean_embeddings(model)
    seed = 123
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    model.load_state_dict(torch.load('ckpts/reddit.pt'))

    print("="*30)
    print("="*30)
    print("Seed: {}".format(seed))
    print("="*30)     
    print("="*30)

    
    trainer = get_trainer(dataset,model,mean_embeds,seed)
    print("Baseline class 0: {}".format(trainer[0].quantatitive_baseline()))
    print("Baseline class 1: {}".format(trainer[1].quantatitive_baseline()))

if __name__ == "__main__":
    #These are all functions used to gain insight into the dataset
    #Also used for tuning hyperparameters and getting results
    #Uncomment them to use them
    
    #accuracy_f1()
    #test_GCN() #Not recommended, takes ~ 3 hours
    test_interpreter(show=True)
    #info_dataset()
    #view_data()
    #get_baseline() # Random baseline
