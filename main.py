import yaml
import argparse
import os
import torch
import wandb

def main(args):
    """
    Main entry point for running anomaly detection experiments.
    """
    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully:")
    print(yaml.dump(config, default_flow_style=False))

    # --- Setup Device ---
    # Override device to CPU if model is ARIMA, as it's CPU-bound
    if config['model']['name'] == 'ARIMA':
        config['system']['device'] = 'cpu'
        
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    config['system']['device'] = device

    # --- Dynamically Select and Run Experiment ---
    model_name = config['model']['name']
    
    if model_name == 'AnomalyCCANN':
        from src.pipelines.swat_pipeline import run_experiment
    elif model_name == 'ARIMA':
        from src.pipelines.arima_pipeline import run_experiment
    elif model_name == 'LSTM_VAE':
        from src.pipelines.lstm_vae_pipeline import run_experiment
    elif model_name == 'LSTM_AE':
        from src.pipelines.lstm_ae_pipeline import run_experiment
    elif model_name == 'TRANSFORMER_AE':
        from src.pipelines.transformer_ae_pipeline import run_experiment
    elif model_name == 'GAT':
        from src.pipelines.gat_pipeline import run_experiment
    elif model_name == 'GDN':
        from src.pipelines.gdn_pipeline import run_experiment
    elif model_name == 'TopoGDN':
        from src.pipelines.topo_gdn_pipeline import run_experiment
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Start Experiment ---
    wandb.init(
        project=config.get('project_name', 'default-project'),
        entity=config.get('entity'),
        name=config.get('experiment_name', 'default-experiment'),
        config=config
    )
    
    run_experiment(config)

    wandb.finish()

if __name__ == '__main__':
    # Add a check for the config file to prevent errors
    parser = argparse.ArgumentParser(description="Anomaly Detection with Topological Deep Learning")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit()
    main(args) 