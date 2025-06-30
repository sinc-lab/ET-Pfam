"""
This script tests all models in the specified directory using both a centered 
window and a sliding window. It aggregates the results and saves them into a CSV file.
Parameters:
    -m, --models_folder: Path to the folder containing the models.
Usage example:
    python3 test_base.py -m models/mini/
"""

import os
import sys
import warnings
import argparse
import torch as tr
from src.model import BaseModel
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test
from src.utils import load_config, ResultsTable

# Filter some warnings
warnings.filterwarnings("ignore", 
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)

def parser():
    parser = argparse.ArgumentParser(description="Run tests on all models in a directory.")
    parser.add_argument("-m", "--models_folder", type=str, required=True, 
                        help="Path to the folder containing the models.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    models_folder = args.models_folder

    # Check if the models folder exists
    if not os.path.exists(models_folder):
        print(f"Error: The specified models folder '{models_folder}' does not exist.")
        sys.exit(1)

    # Initialize results table
    results = ResultsTable(is_ensemble=False)

    width = os.get_terminal_size().columns
    
    # Iterate over all models in the models folder
    for model_name in os.listdir(models_folder):
        path = os.path.join(models_folder, model_name) 

        # Skip non-directory entries
        if not os.path.isdir(path):
            continue
        
        # Load config and model weights
        config = load_config(f"{path}/config.json")
        filename = f"{path}/weights.pk"

        categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]

        # Load the model
        model = BaseModel(len(categories), lr=config['lr'], device=config['device'])
        model.load_state_dict(tr.load(filename))
        model.eval()
            
        # Evaluate the model using centered window and sliding window tests
        print("\nTesting centered window for model:", model_name)
        CwS = centered_window_test(config, model, path)

        print("\nTesting sliding window for model:", model_name)
        _, SwA, SwC = sliding_window_test(config, model, path)

        # Add results to the table
        results.add_entry(model_name, CwS, SwA, SwC)

        print("\n" + ">" * width)
    
    # Save results to a CSV file
    path = f"results/{config['dataset']}/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_file = f"{path}base_metrics.csv"
    results.save(results_file)