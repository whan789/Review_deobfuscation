import pandas as pd
import os

def data_load(data_path):
    path = data_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test CSV file not found at {path}")
    test = pd.read_csv(path, encoding='utf-8-sig')
    return test
    



