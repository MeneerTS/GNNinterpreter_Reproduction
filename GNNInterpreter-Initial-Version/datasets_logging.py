import random
import numpy as np
import torch
import os

# Import logging methods for each dataset
from mutag_explanation import mutaglog
from motif_explanation import motiflog
from shape_explanation import shapelog
from cyclicity_explanation import cyclicitylog
from xgnn_explanation import xgnnlog

def main():
    # Get 25 random seeds
    seeds = list(range(1))

    # Create 'logs' directory (if it doesn't exist)
    logs_directory = "logs"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    # XGNN experiments
    xgnnlog(seeds)

    # MUTAG experiments
    # mutaglog(seeds)

    # Motif experiments
    # motiflog(seeds)

    # Shape experiments
    # shapelog(seeds)

    # Cyclicity experiments
    # cyclicitylog(seeds)


if __name__ == '__main__':
    main()
