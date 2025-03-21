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

## Dataset

The SWAT dataset should be placed in the `data/SWAT` directory with:
- `SWATv0_train.csv`: Training data (normal operation)
- `SWATv0_test.csv`: Test data (includes attacks)

Note: The dataset is not included in this repository due to size constraints.

## Usage

For testing purposes, use a sample rate of 0.1 (10% of data) due to the large dataset size:

```bash
python swat_anomaly_detection.py
```

Modify the sample rate in `swat_anomaly_detection.py`:

```python
train_data, test_data = load_swat_data(train_path, test_path, sample_rate=0.1) 
```

For full dataset training:

```bash
nohup python swat_anomaly_detection.py > swat_output.log 2>&1 &
```

## Model Architecture

TopoGuard uses a Combinatorial Complex Attention Neural Network with:
- HMC layers for message passing across different topological dimensions
- An encoder-decoder architecture for anomaly detection
- Component-level anomaly localization capability