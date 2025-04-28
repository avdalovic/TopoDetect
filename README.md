# TopoGuard: Topological Deep Learning for Anomaly Detection in Industrial Control Systems

This research project implements a novel approach to anomaly detection in industrial control systems using topological deep learning techniques. The system is evaluated on the SWAT (Secure Water Treatment) dataset.

## Environment Setup

The project uses Python 3.11 with the following dependencies:

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

The script expects the SWAT dataset CSV files (`SWATv0_train.csv`, `SWATv0_test.csv`) to be located in a directory specified by the `--data_dir` argument (default: `data/SWAT`).

## Usage

The main script is `swat_anomaly_detection_three_levels.py`. You can run it from the command line.

**Basic Usage (using defaults):**
```bash
python swat_anomaly_detection_three_levels.py
```

**Example with Custom Parameters:**
```bash
python swat_anomaly_detection_three_levels.py \
    --data_dir /path/to/your/swat/data \
    --output_base_dir results/experiment1 \
    --sample_rate 0.01 \
    --epochs 20 \
    --lr 0.002 \
    --threshold_percentile 99.0 \
    --use_gpu
```

**Command-Line Arguments:**

*   `--data_dir` (str): Directory containing SWAT CSV files (default: `data/SWAT`).
*   `--output_base_dir` (str): Base directory for saving runs, models, and results (default: `runs`). A timestamped sub-folder will be created within this directory for each run.
*   `--sample_rate` (float): Fraction of data to sample (0.0 to 1.0, default: 0.001).
*   `--validation_split` (float): Fraction of training data to use for validation (default: 0.2).
*   `--lr` (float): Learning rate (default: 0.005).
*   `--epochs` (int): Maximum number of training epochs (default: 10).
*   `--weight_decay` (float): Weight decay (L2 penalty) (default: 1e-5).
*   `--grad_clip` (float): Gradient clipping value (default: 1.0).
*   `--seed` (int): Random seed for reproducibility (default: 42).
*   `--use_gpu` (flag): Use GPU if available (default: False).
*   `--threshold_method` (str): Method for threshold calibration ('percentile' or 'mean_sd', default: 'percentile').
*   `--threshold_percentile` (float): Percentile for thresholding (if method is percentile, default: 99.0).
*   `--sd_multiplier` (float): Multiplier for standard deviation (if method is mean_sd, default: 2.5).
*   `--patience` (int): Epochs to wait for improvement before early stopping (default: 3).
*   `--min_delta` (float): Minimum change in F1 score to qualify as improvement for early stopping (default: 1e-4).

Use `python swat_anomaly_detection_three_levels.py --help` to see all options.

## Output

The script will:
*   Print training progress, loss, validation threshold calibration, and test set evaluation results to the console.
*   Create a timestamped run directory inside `--output_base_dir`.
*   Inside the run directory:
    *   `data/`: Saves the processed validation and test data CSVs.
    *   `model_checkpoints/`: Saves model checkpoints (`.pt` files) for each epoch and the best performing model based on test F1 (`best_test_ep.pt`).
    *   `results/`: Saves attack detection results (`attack_detection.npz`) if analysis runs successfully.
*   Generate a `hierarchical_localization_log.txt` file in the main project directory containing detailed anomaly localization information for relevant samples.

## Model Architecture

TopoGuard uses a Combinatorial Complex Attention Neural Network with:
- HMC layers for message passing across different topological dimensions
- An encoder-decoder architecture for anomaly detection
- Component-level anomaly localization capability