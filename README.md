# TopoDetect: A Topological Deep Learning Framework for Anomaly Detection in ICS Networks

This research project implements a novel approach to anomaly detection in industrial control systems using topological deep learning techniques. The system is evaluated on the SWAT (Secure Water Treatment), WADI (Water Distribution), and TEP (Tennessee Eastman Process) datasets.

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

### GPU Setup

**Check your CUDA version first:**
```bash
nvidia-smi
```

**Install system dependencies (required for PyTorch geometric):**
```bash
sudo apt update
sudo apt install build-essential
```

**Select the appropriate CUDA version for PyTorch:**
- CUDA 12.1+ → use `cu121`
- CUDA 11.8 → use `cu118`  
- CUDA 11.7 → use `cu117`
- No GPU → use `cpu`

**Example for CUDA 12.4 (RTX A6000):**
```bash
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

**Verify GPU setup:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

**Troubleshooting:**
- If you get `g++` compiler errors, install build tools: `sudo apt install build-essential`
- If CUDA versions don't match, adjust the `cu###` suffix accordingly
- For compilation issues, you may also need: `sudo apt install cmake ninja-build`

Alternatively, you can install all dependencies using:

```bash
pip install -r requirements.txt
```
Ensure you install the correct PyTorch version for your system (CPU or GPU). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Datasets

The project supports three industrial control system datasets:

### SWAT (Secure Water Treatment)
- Files: `SWATv0_train.csv`, `SWATv0_test.csv`
- Location: `data/SWAT/`
- 51 sensors and actuators across 6 process stages

### WADI (Water Distribution)
- Files: `WADI_train.csv`, `WADI_test.csv`
- Location: `data/WADI/`
- 127 sensors and actuators across 5 subsystems

### TEP (Tennessee Eastman Process)
- Files: `TEP_train.csv`, `TEP_test_cons_p2s_s1.csv`
- Location: `data/TEP/`
- 53 sensors and actuators in chemical process simulation

Each dataset should be placed in its respective directory as specified in the configuration file.

## Usage

The main entry point for all experiments is `main.py`. The behavior of the script is controlled by a configuration file.

**Basic Usage:**
```bash
# Run SWAT experiment
python main.py --config config.yaml

# Run WADI experiment
python main.py --config config_wadi.yaml

# Run TEP experiment (when implemented)
python main.py --config config_tep.yaml
```

The configuration file centralizes all hyperparameters and settings for the experiment, including:
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
├── config.yaml           # SWAT experiment configuration
├── config_wadi.yaml      # WADI experiment configuration
├── configs/              # Additional experiment configurations
├── data/                 # Raw and processed data
│   ├── SWAT/            # SWAT dataset files
│   ├── WADI/            # WADI dataset files
│   └── TEP/             # TEP dataset files
├── main.py               # Main script to run experiments
├── README.md
├── requirements.txt
├── saved_test_data/      # Cached test data to speed up runs
├── src/                  # Source code
│   ├── datasets/         # Dataset classes (SWaTDataset, WADIDataset, etc.)
│   ├── models/           # Model architectures (AnomalyCCANN, etc.)
│   ├── pipelines/        # Dataset-specific pipelines (swat_pipeline, wadi_pipeline)
│   ├── trainers/         # Training and evaluation logic with wandb integration
│   └── utils/            # Utility functions (topology builders, attack utils)
└── wandb/                # Wandb local logs
```

## Model Architecture

TopoDetect uses a Combinatorial Complex Attention Neural Network (AnomalyCCANN) with:
- HMC layers for message passing across different topological dimensions (0-cells, 1-cells, 2-cells)
- An encoder-decoder architecture for anomaly detection
- Component-level anomaly localization capability
- Support for both reconstruction and temporal prediction modes

## Features

- **Multi-Dataset Support**: SWAT, WADI, and TEP datasets with dataset-specific preprocessing
- **Comprehensive Logging**: Integration with Weights & Biases for experiment tracking
- **Advanced Metrics**: Time-aware precision, recall, and F1 scores (eTaP, eTaR, eTaF1)
- **Visualization**: Attack detection timelines, reconstruction error plots, and localization maps
- **Flexible Configuration**: YAML-based configuration system for easy experiment management
- **Attack Analysis**: Individual attack scenario analysis with detailed performance metrics