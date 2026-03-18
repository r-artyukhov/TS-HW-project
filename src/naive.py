import numpy as np
import pandas as pd


def naive_forecast(train_df, test_df, horizon):
    test_df = test_df.copy()
    test_df['naive'] = np.nan

    for uid in train_df['Commodity'].unique():
        last_value = train_df[train_df['Commodity'] == uid].sort_values('Date')['Average'].iloc[-1]

        idx = test_df[test_df['Commodity'] == uid].index

        test_df.loc[idx, 'naive'] = last_value

    return test_df