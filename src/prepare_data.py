import numpy as np
import pandas as pd


def prepare_data(df_new):

    df_new = df_new.drop(['SN', 'Unit', 'Minimum', 'Maximum'], axis=1)
    
    df_filtered = df_new.groupby('Commodity').filter(lambda x: len(x) >= 2500)
    df_filtered = df_filtered.sort_values(by='Date')
    
    df = df_filtered.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    start_date = "2013-06-16"
    end_date   = "2021-05-13"

    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_regular = pd.concat(
    [process_series(g, full_dates) for _, g in df.groupby('Commodity')],
    ignore_index=True)
    
    train, test_30 = split_df(df_regular, horizon=30)
    test_14, _ = split_df(test_30, horizon=16)
    test_1, _ = split_df(test_14, horizon=13)

    return train, test_1, test_14, test_30

def process_series(group, full_dates):
    group = group.sort_values('Date')
    group = group.set_index('Date')
    group = group.reindex(full_dates)
    group['Commodity'] = group['Commodity'].iloc[0]
    group['Average'] = group['Average'].ffill()

    return group.reset_index().rename(columns={'index': 'Date'})
    
def split_df(df, horizon):
    train_list, test_list = [], []

    for uid, group in df.groupby('Commodity'):
        group = group.sort_values('Date').reset_index(drop=True)
        train = group.iloc[:-horizon]
        test  = group.iloc[-horizon:]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df  = pd.concat(test_list, ignore_index=True)

    return train_df, test_df
