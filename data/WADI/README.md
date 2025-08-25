# WADI Dataset Setup

The WADI (Water Distribution) dataset contains readings across 127 features (sensors, actuators) for a water distribution system.

## Dataset Request

**⚠️ IMPORTANT: This dataset requires formal request and approval.**

1. **Request dataset from:** https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
2. **Fill out the request form** and wait for approval
3. **Download** the approved dataset files

## Dataset Processing

After receiving approval and downloading:

1. **Download these files** from the first WADI dataset (WADI.A1_9 Oct 2017) into this directory:
   - `WADI_attackdata.csv`
   - `WADI_14days.csv`

2. **Process the data:** Run the processing script:
   ```bash
   python3 process_WADI.py
   ```

## Final Files Required

After processing, you should have these files in this directory:
- `WADI_train.csv` - Training data
- `WADI_test.csv` - Test data

**Note:** The original downloaded CSV files can be removed after successful processing.



