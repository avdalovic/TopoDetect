# TopoDetect: A Topological Deep Learning Framework for Anomaly Detection in ICS Networks

This research project implements a novel approach to anomaly detection in industrial control systems using topological deep learning techniques. The system is evaluated on the SWAT (Secure Water Treatment), WADI, and TEP datasets.

## Environment Setup

The project uses Python 3.11. The recommended setup is to use a `conda` environment.

```bash
# Create and activate conda environment
conda create -n topox python=3.11
conda activate topox

# Install base packages
conda install -y numpy scipy matplotlib pandas==1.5.3 networkx jupyter

# Install topological libraries
pip install toponetx
pip install topomodelx

# Install PyTorch with CUDA support (replace ${CUDA} with your CUDA version like cu117)
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```

Alternatively, you can install all dependencies using:

```bash
pip install -r requirements.txt
```
Ensure you install the correct PyTorch version for your system (CPU or GPU). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Dataset

The project expects the dataset CSV files (e.g., `SWATv0_train.csv`, `SWATv0_test.csv`) to be located in the directory specified in the configuration file (e.g., `data/SWAT`).

## Usage

The main entry point for all experiments is `main.py`. The behavior of the script is controlled by a configuration file.

**Basic Usage:**
```bash
python main.py --config config.yaml
```

The `config.yaml` file centralizes all hyperparameters and settings for the experiment, including:
- System settings (device, checkpoint directory)
- Data parameters (dataset name, paths, sampling rates)
- Model architecture (layer configurations, temporal mode)
- Training parameters (learning rate, batch size, epochs)
- Anomaly detection logic (thresholding methods)
- `wandb` logging details (project, entity, experiment name)

You can create different `.yaml` files in the `configs/` directory to define various experiments.

## Output

The script will:
*   Print configuration and progress to the console.
*   Log all metrics, results, and the configuration to [Weights & Biases (wandb)](https://wandb.ai).
*   Create a checkpoint directory (e.g., `model_checkpoints/swat_static_ccann/`) for each experiment.
*   Inside the checkpoint directory:
    *   Saves model checkpoints (`.pt` files) for each epoch.
    *   Saves the best performing model based on the validation set.
    *   Saves final test results and other artifacts as `.pkl` files.

## Project Structure

```
TDS_ICS_project/
├── configs/              # Experiment configuration files (.yaml)
├── data/                 # Raw and processed data
│   ├── SWAT/
│   └── ...
├── main.py               # Main script to run experiments
├── README.md
├── requirements.txt
├── saved_test_data/      # Cached test data to speed up runs
├── src/                  # Source code
│   ├── datasets/         # Dataset classes (e.g., SWaTDataset)
│   ├── models/           # Model architectures (e.g., AnomalyCCANN)
│   ├── pipelines/        # Experiment-specific pipelines (e.g., swat_pipeline)
│   ├── trainers/         # Training and evaluation logic
│   └── utils/            # Utility functions (e.g., topology builders)
└── wandb/                # Wandb local logs
```

## Model Architecture

TopoGuard uses a Combinatorial Complex Attention Neural Network with:
- HMC layers for message passing across different topological dimensions
- An encoder-decoder architecture for anomaly detection
- Component-level anomaly localization capability