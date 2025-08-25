# TMS-ICS Artifact

## Quick Start (5 minutes)

**Prerequisites:**
- SWAT dataset files in `data/SWAT/` (see main README.md for data request instructions)
- Python environment with dependencies installed

**Quick functionality test:**
```bash
conda activate topox
python main.py --config configs/tms_ics_swat_quick.yaml
```

## Paper Results Reproduction

**Table I - Detection Performance:**
```bash
# Run our TMS-ICS method (10% data)
python main.py --config configs/tms_ics_swat_table1.yaml
```

**Baseline comparisons:** Available at the existing artifact: https://zenodo.org/records/15120036

**Localization experiments:** Results calculated using `localization_map.txt`

## Key Artifacts Generated

- `checkpoints/[experiment_name]/final_results.pkl` - Final results
- `checkpoints/[experiment_name]/metrics.json` -  Metrics
- `checkpoints/[experiment_name]/best_model.pt` - Model weights

## Troubleshooting

1. **Dataset not found**: Ensure SWAT dataset files are in `data/SWAT/`
2. **CUDA errors**: Set `device: "cpu"` in config if no GPU available

