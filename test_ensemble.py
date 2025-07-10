"""
This script runs ensemble evaluation on a set of models using various voting strategies.
It performs both centered window and sliding window testing, and saves the 
results into a CSV file.
Parameters:
    -v, --voting_strategy: Voting strategy to use (e.g., 'simple_voting', 'score_voting', 
                          'weighted_model', 'weighted_families', 'all').
    -m, --models_path: Path to the directory containing the models to ensemble.
    -c, --config_path: Path to the configuration file (.json).
    -w, --ensemble_weights_path: Path to the model weights for weighted voting strategies.
    -o, --output_path: Path to save the results.
    -e, --exp_name: Experiment name for saving ensemble weights.
    -p, --partition: Dataset partition to test on (default: 'test').
Usage example:
    python3 test_ensemble.py -v all -m models/mini/ (to train ensemble)
    python3 test_ensemble.py -v all -m models/mini/ -w models/mini/ (to use pre-trained ensemble weights)
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
    parser.add_argument("-w", "--ensemble_weights_path", type=str, required=False,
                        help="Path (folder) to the model weights for weighted voting strategies")
    parser.add_argument("-p", "--partition", type=str, required=False,
                        help="Dataset partition to test on (default: 'test')",
                        default='test')     
    parser.add_argument("-o", "--output_path", type=str, required=False,
                        help="Path to save the results",
                        default=None)
    parser.add_argument("-e", "--exp_name", type=str, required=False,
                        help="Experiment name for saving ensemble weights",
                        default=None)
    args = parser.parse_args()
    return args

def run_ensemble_tests(models_path, config, voting_strategy, output_path,
                       ensemble_weights_path=None, exp_name=None, partition='test'):
    """
    Initializes the ensemble, fits it (if required), and runs both sliding and 
    centered window tests.
        Args:
        models_path (str): Path to ensemble model directories.
        config (dict): Configuration dictionary.
        voting_strategy (str): Selected voting strategy.
        output_path (str): Directory to store results.
        ensemble_weights_path (str, optional): Path for ensemble weights.
        exp_name (str, optional): Experiment name for saving ensemble weights.
        partition (str, optional): Dataset partition to test on (default: 'test').
    """
    os.makedirs(output_path, exist_ok=True)

    width = os.get_terminal_size().columns
    
    # Initialize the ensemble
    print("\n" + ">" * width)
    print(f"\nInitializing ensemble with voting strategy: {voting_strategy}")
    if ensemble_weights_path:
        # Use the provided ensemble weights if specified
        print(f"Using model weights from: {ensemble_weights_path}")
        ensemble = EnsembleModel(models_path, config,
                                 voting_strategy, 
                                 ensemble_weights_path=ensemble_weights_path, 
                                 exp_name=exp_name)
    else:
        # Fit the ensemble if no weights are provided
        print("Fitting ensemble model...")
        ensemble = EnsembleModel(models_path, config, voting_strategy, exp_name=exp_name)
        ensemble.fit()

    # Test the ensemble with centered and sliding window methods
    print("\n" + "-" * width)
    print("\nRunning centered window test...")
    CwS = centered_window_test(config, ensemble, output_path, is_ensemble=True,
                         voting_strategy=voting_strategy, partition=partition)

    print("\n" + "-" * width)
    print("\nRunning sliding window test...")
    _, SwA, SwC = sliding_window_test(config, ensemble, output_path, is_ensemble=True,
                                       partition=partition)

    return CwS, SwA, SwC

if __name__ == "__main__":
    args = parser()

    valid_strategies = ['weighted_families','simple_voting', 'score_voting', 
                        'weighted_model', 'all']
    if args.voting_strategy not in valid_strategies:
        raise ValueError(f"Invalid voting strategy: {args.voting_strategy}. " \
                         f"Choose from {valid_strategies[:-1]} or 'all'.")

    config = load_config(args.config_path)

    if args.output_path:
        output_path = args.output_path
    else:
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
                                               ensemble_weights_path=args.ensemble_weights_path,
                                               exp_name=args.exp_name,
                                               partition=args.partition)
            results.add_entry(strategy, CwS, SwA, SwC)
    else:
        path = os.path.join(output_path, args.voting_strategy)
        CwS, SwA, SwC = run_ensemble_tests(models_path=args.models_path, 
                                           config=config,
                                           voting_strategy=args.voting_strategy,
                                           output_path=path,
                                           ensemble_weights_path=args.ensemble_weights_path,
                                           exp_name=args.exp_name,
                                           partition=args.partition)
        results.add_entry(args.voting_strategy, CwS, SwA, SwC)

    results_file = os.path.join(output_path, "ensemble_metrics.csv")
    results.save(results_file)