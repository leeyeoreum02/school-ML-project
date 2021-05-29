import pandas as pd
import os


def load_titanic_data(mode: str) -> pd.DataFrame:
    TRAIN_PATH = os.path.join('data', 'train.csv')
    TEST_PATH = os.path.join('data', 'test.csv')
    modes = {'train': TRAIN_PATH, 'test': TEST_PATH}
    csv_path = os.path.join(modes[mode])
    return pd.read_csv(csv_path)