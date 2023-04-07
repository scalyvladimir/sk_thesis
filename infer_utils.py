import pandas as pd

rows = dict.fromkeys(['philips3', 'siemens3', 'ge3', 'philips15', 'siemens15', 'ge15'])

def create_df():
    return pd.DataFrame(rows, columns=rows)