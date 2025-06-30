"""
This script runs ensemble evaluation on a set of models using various voting strategies.
It performs both centered window and sliding window testing, and saves the 
results into a CSV file.
Parameters:
    -v, --voting_strategy: Voting strategy to use (e.g., 'simple_voting', 'score_voting', 
                          'weighted_model', 'weighted_families', 'all').
    -m, --models_path: Path to the directory containing the models to ensemble.
    -c, --config_path: Path to the configuration file (.json).
    -w, --model_weights_path: Path to the model weights for weighted voting strategies.
Usage example:
    python3 test_ensemble.py -v all -m models/full/ (to train ensemble)
    python3 test_ensemble.py -v all -m models/full/ -w models/full/ (to use pre-trained ensemble weights)
"""
import os
import argparse
import torch as tr
from src.ensemble import EnsembleModel
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test
from src.utils import load_config, ResultsTable

tr.multiprocessing.set_sharing_strategy('file_system')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--voting_strategy", type=str, required=False, 
                        help="Voting strategy: 'simple_voting', 'score_voting', " \
                        "'weighted_model', 'weighted_families', 'all'",
                        default="simple_voting")
    parser.add_argument("-m","--models_path", type=str, required=True, 
                        help="Path to the models to ensemble",
                        default="models/mini/")
    parser.add_argument("-c","--config_path", type=str, required=False,
                        help="Path to the config file (.json)",
                        default="config/base.json") 
    parser.add_argument("-w", "--model_weights_path", type=str, required=False,
                        help="Path (folder) to the model weights for weighted voting strategies")
    args = parser.parse_args()
    return args

def run_ensemble_tests(models_path, config, voting_strategy, output_path,
                       model_weights_path=None):
    """
    Initializes the ensemble, fits it (if required), and runs both sliding and 
    centered window tests.
        Args:
        models_path (str): Path to ensemble model directories.
        config (dict): Configuration dictionary.
        voting_strategy (str): Selected voting strategy.
        output_path (str): Directory to store results.
        model_weights_path (str, optional): Path for ensemble weights.
    """
    emb_path = config['emb_path']
    data_path = config['data_path']
    cat_path = os.path.join(data_path, "categories.txt")

    os.makedirs(output_path, exist_ok=True)
    
    # Initialize the ensemble
    print(f"Initializing ensemble with voting strategy: {voting_strategy}")
    if model_weights_path:
        # Use the provided ensemble weights if specified
        print(f"Using model weights from: {model_weights_path}")
        ensemble = EnsembleModel(models_path, emb_path, data_path, cat_path, 
                           voting_strategy, model_weights_path=model_weights_path)
    else:
        # Fit the ensemble if no weights are provided
        print("Fitting ensemble model...")
        ensemble = EnsembleModel(models_path, emb_path, data_path, cat_path, voting_strategy)
        ensemble.fit()

    # Test the ensemble with centered and sliding window methods
    print("Running centered window test...")
    CwS = centered_window_test(config, ensemble, output_path, is_ensemble=True,
                         voting_strategy=voting_strategy)

    print("Running sliding window test...")
    _, SwA, SwC = sliding_window_test(config, ensemble, output_path)

    return CwS, SwA, SwC

if __name__ == "__main__":
    args = parser()

    valid_strategies = ['simple_voting', 'score_voting', 'weighted_model', 
                        'weighted_families', 'all']
    if args.voting_strategy not in valid_strategies:
        raise ValueError(f"Invalid voting strategy: {args.voting_strategy}. " \
                         f"Choose from {valid_strategies[:-1]} or 'all'.")

    config = load_config(args.config_path)

    output_path = f'results/{config["dataset"]}/'
    os.makedirs(output_path, exist_ok=True)
    
    results = ResultsTable(is_ensemble=True)

    if args.voting_strategy == 'all':
        for strategy in valid_strategies[:-1]:  # Exclude 'all' 
            path = os.path.join(output_path, strategy)
            CwS, SwA, SwC = run_ensemble_tests(models_path=args.models_path, 
                                               config=config,
                                               voting_strategy=strategy,
                                               output_path=path,
                                               model_weights_path=args.model_weights_path)
            results.add_entry(strategy, CwS, SwA, SwC)
    else:
        path = os.path.join(output_path, args.voting_strategy)
        CwS, SwA, SwC = run_ensemble_tests(models_path=args.models_path, 
                                           config=config,
                                           voting_strategy=args.voting_strategy,
                                           output_path=path,
                                           model_weights_path=args.model_weights_path)
        results.add_entry(args.voting_strategy, CwS, SwA, SwC)

    results_file = os.path.join(output_path, "ensemble_metrics.csv")
    results.save(results_file)