from gnninterpreter import *
import random
from collections import defaultdict, namedtuple
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import re
# import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px

from tqdm.auto import tqdm, trange

import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score
import time
import os

def calculate_average_train_time(file_path):
    total_train_time = 0
    num_entries = 0

    try:
        # Open the file
        with open(file_path, 'r') as file:
            # Skip the header
            next(file)
            # Read each line in the file
            for line in file:
                if line[0] == 'S':
                    continue
                # Split the line into columns
                train_time_str = line.split('\t')[-1]
                # Convert train time to float and accumulate
                total_train_time += float(train_time_str)
                num_entries += 1

        # Calculate the average train time
        if num_entries > 0:
            average_train_time = total_train_time / num_entries
            return total_train_time, average_train_time
        else:
            return None
    except FileNotFoundError:
        print("File not found:", file_path)
        return None

def parse_array(array_string):
    return [float(x) if '.' in x or 'e' in x.lower() else int(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+\-]?\d+)?', array_string)]


def calculate_average_acc(log_path):
    '''
    use: give path of txt for acc

    back: prints df of results per class and returns it
    '''

    try:
        # Fix to reading shape and motif logs
        if log_path[-5] == "e" or log_path[-5] == "f": #shape or motif
            lines = []
            with open(log_path, 'r') as f:
                cur_line = ""
                for i, line in enumerate(f):
                    line = line.strip("\n")
                    if i == 0:

                        continue
                    elif line[2] != ".":
                        if i != 1:
                            lines.append(cur_line)
                        cur_line = line
                    else:
                        cur_line = cur_line + line
                lines.append(cur_line)
        else:
            lines = []
            with open(log_path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip("\n")
                    if i == 0:
                        continue
                    lines.append(line)

        split_lines = [l.split("\t") for l in lines]

        # Create df
        df = pd.DataFrame(split_lines, columns=['Seed', 'Class', 'Mean', 'Std'])

        # Remove brackets
        df['Mean'] = df['Mean'].str.strip('[]')
        df['Std'] = df['Std'].str.strip('[]')

        # Get number of seeds
        df["Seed"] = df['Seed'].apply(pd.to_numeric)
        num_seeds = df['Seed'].max() + 1

        # Get number of classes
        df['Class'] = df['Class'].apply(pd.to_numeric)
        num_classes = df['Class'].max()

        # Calculate expected number of graphs
        # Mutag starts from class 0
        if log_path[-5] == "g" or log_path[-5] == "y":
            num_classes += 1

        expected_num_graphs = num_seeds
        print("Expected number of graphs per class: ", expected_num_graphs)

        # Remove empty graphs if their log generated
        df_non_empty = df.loc[df['Mean'] != "Empty"].copy()

        # Add additional column to motif and shape for the probability of random class
        if log_path[-5] == "e" or log_path[-5] == "f":
            num_classes += 1

        # Create the column labels for each class
        mean_columns = [f'Mean_{i}' for i in range(num_classes)]
        std_columns = [f'Std_{i}' for i in range(num_classes)]

        # Split mean and std into classes with different columns
        df_non_empty[mean_columns] = df_non_empty['Mean'].str.split(expand=True)
        df_non_empty[std_columns] = df_non_empty['Std'].str.split(expand=True)

        # Convert mean and std to numbers
        df_non_empty[mean_columns] = df_non_empty[mean_columns].apply(pd.to_numeric)
        df_non_empty[std_columns] = df_non_empty[std_columns].apply(pd.to_numeric)

        # Group by target class
        dfs_by_class = {class_: group for class_, group in df_non_empty.groupby('Class')}

        for class_num, df_class in dfs_by_class.items():
            # Calculate the num of empty graphs
            actual_num_graphs_in_class = len(df_class)
            num_empty_graphs_in_class = expected_num_graphs - actual_num_graphs_in_class
            print(f"Actual number of graphs in Class {class_num}: {actual_num_graphs_in_class}")
            print(f"Number of empty graphs in Class {class_num}: {num_empty_graphs_in_class}\n")

            # Get columns corresponding to target class
            mean_column = "Mean_" + str(class_num)
            std_column = "Std_" + str(class_num)

            # Get averaged means of target class
            class_means = df_class[mean_column]
            # Add 0's for the removed empty graph rows
            class_mean = np.mean(class_means.tolist() + [0] * (num_empty_graphs_in_class))

            print(f"Averaged Means of Class {class_num}: {class_mean:.3g}")
            # Get averaged stds of target class
            class_stds = df_class[std_column]
            # Add 0's for the removed empty graph rows
            class_std= np.mean(class_stds.tolist() + [0] * (num_empty_graphs_in_class))

            print(f"Averaged Stds of Class {class_num}: {class_std:.3g}\n")

            # Get the number of class prob 1 graphs
            num_perfect_graphs = (df_class[mean_column] > 0.99).sum(axis=0)
            print(f"Number of graphs with class probability 1 for class {class_num}: {num_perfect_graphs}")
            print(df_class.loc[df_class[mean_column] > 0.99])

            # Get the number of "good" graphs, higher than 0.9 target probability
            num_good_graphs = (df_class[mean_column] > 0.9).sum(axis=0)
            print(f"Number of good graphs for class {class_num}: {num_good_graphs}")

            # Get the number of "bad" graphs, less than random probs
            num_bad_graphs = (df_class[mean_column] < 0.1).sum(axis=0)
            print(f"Number of bad graphs for class {class_num}: {num_bad_graphs}\n")

            best_graph = df_class.loc[df_class[mean_column] == df_class[mean_column].max()]
            print(f"Best graphs\n {best_graph}")
            worst_graph = df_class.loc[df_class[mean_column] == df_class[mean_column].min()]
            print(f"Worst graphs\n {worst_graph}")

    except FileNotFoundError:
        print("File not found:", log_path)
        return None

def calculate_average_acc_xgnn(log_path):
    '''
    use: give path of txt for acc

    back: prints df of results per class and returns it
    '''

    try:
        lines = []
        with open(log_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip("\n")
                if i == 0 or line[0] == 'S':
                    continue
                lines.append(line)

        split_lines = [l.split("\t") for l in lines]

        # Create df
        df = pd.DataFrame(split_lines, columns=['Seed', 'Class', 'Probabilities'])

        # Convert Probabilities column to numeric
        df['Probabilities'] = pd.to_numeric(df['Probabilities'])

        # Get stats  per class
        stats_per_class = df.groupby('Class')['Probabilities'].agg(['mean', 'std'])
        print(stats_per_class)

    except FileNotFoundError:
        print("File not found:", log_path)
        return None
    
def main():
    # Accuracy
    accuracy_paths = ["logs/class_probs_xgnn.txt", "logs/class_probs_motif.txt", "logs/class_probs_shape.txt", "logs/class_probs_mutag.txt"]
    
    for path in accuracy_paths:
        print(path[5:-4].upper())
        if 'xgnn' in path:
            calculate_average_acc_xgnn(path)
        else:
            calculate_average_acc(path)
    
    # Training time
    training_paths = ["logs/train_times_xgnn.txt", "logs/train_times_motif.txt", "logs/train_times_shape.txt", "logs/train_times_mutag.txt"]

    for _, file_path in enumerate(training_paths):
        total_train_time, average_train_time = calculate_average_train_time(file_path)
        if average_train_time is not None:
            datasetName = file_path.split("_")[-1].split(".")[0]
            print(f"Average Train Time {datasetName}: ", average_train_time)
            print(f"Total Train Time{datasetName}: {total_train_time}")


if __name__ == '__main__':
    main()
