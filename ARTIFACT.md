# USENIX Artifact Evaluation - TMS-ICS

This document provides instructions for USENIX Security artifact evaluation.

## Quick Start (5 minutes)

**Prerequisites:**
- SWAT dataset files in `data/SWAT/` (see main README.md for data request instructions)
- Python environment with dependencies installed

**Quick functionality test:**
```bash
# Activate environment
conda activate topox

# Run quick test (1% data, 1 epoch)
python main.py --config configs/tms_ics_swat_quick.yaml
```

**Expected output:**
```
--- Test Results ---
Precision: 0.9849
Recall:    0.6306
F1-Score:  0.7689

============================================================
FINAL RESULTS SUMMARY
============================================================
Precision: 0.9849
Recall:    0.6306
F1-Score:  0.7689
============================================================
```

## Paper Results Reproduction

**For Table I results (10% data):**
```bash
python main.py --config configs/tms_ics_swat_table1.yaml
```

**Key artifacts generated:**
- `checkpoints/[experiment_name]/metrics.json` - Core performance metrics
- `checkpoints/[experiment_name]/model.pth` - Trained model weights
- `checkpoints/[experiment_name]/test_residuals.pkl` - Test residuals for analysis

## Configuration

The system uses YAML configuration files. Key settings:
- `data.sample_rate`: Controls data subset (0.01 = 1%, 0.1 = 10%)
- `training.epochs`: Number of training epochs
- `logging.use_wandb`: Set to `false` for console-only output

## Expected Performance

**Quick test (1% data):**
- Precision: ~0.98
- Recall: ~0.63
- F1-Score: ~0.77

**Full evaluation (10% data):**
- Results should match Table I in the paper

## Troubleshooting

**Common issues:**
1. **Dataset not found**: Ensure SWAT dataset files are in `data/SWAT/`
2. **CUDA errors**: Set `device: "cpu"` in config if no GPU available
3. **Memory issues**: Reduce `batch_size` in config

**Support:** Check main README.md for environment setup and dataset instructions.
