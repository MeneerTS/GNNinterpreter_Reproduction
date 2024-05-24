import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar

from gnn_explain import gnn_explain



explainer = gnn_explain(6, 30,  1, 50)  ####arguments: (max_node, max_step, target_class, max_iters)

explainer.train()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse

# from utils import progress_bar

# from gnn_explain import gnn_explain

# import time
# import torch
# import random
# import numpy as np

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True

# def measureTrainTime(seeds):
#     class_0 = []
#     class_1 = []

#     for _, seed in enumerate(seeds):
#         seed_all(seed)

#         # Class 0
#         explainer = gnn_explain(6, 30, 0, 50)  ####arguments: (max_node, max_step, target_class, max_iters)
        
#         # Train explainer
#         time_class0 = explainer.train(onlyMeasureTime=True)
#         class_0.append(time_class0)

#         # Class 1
#         explainer0 = gnn_explain(6, 30, 1, 50)  
        
#         # Train explainer
#         time_class1 = explainer0.train(onlyMeasureTime=True)
#         class_1.append(time_class1)

#         # print(f"XGNN explainer training time: {time_class1:.4f} sec for class 1 and {time_class0:.4f} sec for class 0.")

#     print(f"Class 0: {class_0}")
#     print(f"Class 1: {class_1}")

#     print(f"Mean train time class 0: {np.mean(class_0)}")
#     print(f"Mean train time class 1: {np.mean(class_1)}")

# def figure_four(seeds):
#     for _, seed in enumerate(seeds):
#         seed_all(seed)

#         # Figure 4 (XGNN paper)
#         # labels: C = 0, N = 1, O = 2
#         # max_nodes = [3, 4, 5, 6, 7] # max number of nodes for explanation graph
#         max_nodes = [5, 6, 7]
#         for _, val in enumerate(max_nodes):
#             # Class 1 (mutagenic)
#             explainer = gnn_explain(max_node=val, max_step=30, target_class=1, max_iters=100)
#             explainer.train()

#         for _, val in enumerate(max_nodes):
#             # Class 0 (non-mutagenic)
#             explainer = gnn_explain(max_node=val, max_step=30, target_class=0, max_iters=100)
#             explainer.train()

# def figure_five(seeds):
#     for _, seed in enumerate(seeds):
#         seed_all(seed)

#         # Figure 5 (XGNN paper)
#         # node_limit = 5
#         max_nodes = [5, 6, 7]
#         labels = [0, 1, 2, 3, 4, 5, 6] # initial molecule to start from
#         for _, val in enumerate(labels):
#             for _, limit in enumerate(max_nodes):
#                 # Class 1 (mutagenic)
#                 explainer = gnn_explain(max_node=limit, max_step=30, target_class=1, max_iters=100)
#                 explainer.train(label=val)


# if __name__ == '__main__':
#     # Training times of XGNN per class (MUTAG dataset)
#     # measureTrainTime(seeds)

#     # Figure 4
#     # figure_four(seeds)

#     # Figure 5
#     # figure_five(seeds)

#     # for _, seed in enumerate(seeds):
#     #     # Set seeds
#     #     torch.manual_seed(seed)
#     #     random.seed(seed)
#     #     np.random.seed(seed)

#     #     Class 0 graphs (non-mutagenic)
#     #     explainer = gnn_explain(6, 30, 0, 50) ####arguments: (max_node, max_step, target_class, max_iters)
#     #     explainer.train()

#     #     Default implementation - start from a single atom of Carbon for explanation graph

