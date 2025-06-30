"""
This script is designed to train and test an individual base model. The test is
performed using a centered and sliding window.
Parameters:
    -c, --config_path: Path to the configuration file (json).
    -o, --output_path: Base output path for results.
Usage example:
    python3 train_and_test_base.py -o models/model1/
"""
import os 
import argparse
import shutil
import torch as tr
from src.model import BaseModel
from src.utils import load_config
from src.train import train
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test

def parser():
    parser = argparse.ArgumentParser(description="Train and test a base model.")
    parser.add_argument("-c", "--config_path", type=str, required=False, 
                        help="Path to the configuration file (JSON).",
                        default="config/base.json")
    parser.add_argument("-o", "--output_path", type=str, required=True, 
                        help="Base output path for results.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

if __name__ == "__main__":
    args = parser()
    config_path = args.config_path
    output_path = args.output_path

    # Copy the config file to the output path
    shutil.copyfile(config_path, os.path.join(output_path, "config.json"))

    # Load the configuration
    config = load_config(config_path)
    categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]

    # Train the model 
    print("Training the model...")
    train(config, categories, output_path)

    # Load the trained model
    model = BaseModel(len(categories), lr=config['lr'], device=config['device'])
    model.load_state_dict(tr.load(f"{output_path}/weights.pk"))
    model.eval()

    # Centered window test
    print("Testing centered window...")
    centered_window_test(config, model, output_path)

    # Sliding window test
    print("Testing sliding window...")
    sliding_window_test(config, model, output_path)