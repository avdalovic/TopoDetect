import pandas as pd
import os

def inspect_csv(file_path):
    """Inspect a CSV file and print first 10 rows"""
    print(f"\nInspecting: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        return
    
    try:
        # Print loading message to debug
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Print file info
        print(f"Shape: {df.shape}")
        
        # Print column names
        print("\nColumn Names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        # Print first 10 rows
        print("\nFirst 10 Rows:")
        print(df.head(10))
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

# File paths
train_file = "WADI_train.csv"
test_file = "WADI_test.csv"

print("Starting inspection...")
inspect_csv(train_file)
inspect_csv(test_file)