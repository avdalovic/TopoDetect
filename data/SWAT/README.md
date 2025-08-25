# SWAT Dataset Setup

The SWAT (Secure Water Treatment) dataset contains data from a 6-stage water treatment process, collected from a water plant testbed in Singapore.

## Dataset Request

**⚠️ IMPORTANT: This dataset requires formal request and approval.**

1. **Request dataset from:** https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
2. **Fill out the request form** and wait for approval
3. **Download** the approved dataset files

## Dataset Processing

After receiving approval and downloading:

1. **Locate the files:** From the first SWAT dataset (SWaT.A1 & A2_Dec 2015), access the `Physical` directory
2. **Find these files:**
   - `SWaT_Dataset_Normal_v1.xlsx`
   - `SWaT_Dataset_Attack_v0.xlsx`
3. **Convert to CSV:** Save these files in this directory as CSV (comma-delimited) files:
   - `SWaT_Dataset_Normal_v1.csv`
   - `SWaT_Dataset_Attack_v0.csv`
4. **Process the data:** Run the processing script:
   ```bash
   python3 process_SWaT.py
   ```

## Final Files Required

After processing, you should have these files in this directory:
- `SWATv0_train.csv` - Training data
- `SWATv0_test.csv` - Test data

**Note:** The original Excel files and intermediate CSV files can be removed after successful processing.
