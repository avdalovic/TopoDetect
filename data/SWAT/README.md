This dataset contains processing for a 6 stage water treatment process, collected from a water plant testbed in Singapore.


Request and download the SWAT dataset from: 
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

- This code is currently only based on the first SWAT dataset (SWaT.A1 & A2_Dec 2015).
- From the first SWAT dataset, access the `Physical` directory and find the following two files:
    - `SWaT_Dataset_Normal_v1.xlsx` and `SWaT_Dataset_Attack_v0.xlsx`
- Save these files in this directory as CSV (comma-delimited) files named `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv` respectively.
- Proceed to run `process_SWaT.py`, which will relabel the files as training/test CSVs:
```sh
python3 process_SWaT.py
