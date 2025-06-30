import yaml
import argparse
import os
import torch
import wandb
from src.pipelines.swat_pipeline import run_experiment

def main():
    """
    Main entry point for running anomaly detection experiments.
    """
    parser = argparse.ArgumentParser(description="Anomaly Detection with Topological Deep Learning")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully:")
    print(yaml.dump(config, default_flow_style=False))

    # --- Setup Device ---
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    config['system']['device'] = device

    # --- Start Experiment ---
    wandb.init(
        project=config['project_name'],
        entity=config['entity'],
        name=config['experiment_name'],
        config=config
    )
    
    run_experiment(config)

    wandb.finish()

if __name__ == '__main__':
    main() 