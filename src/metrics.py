import numpy as np
import pandas as pd


def metrics(df, true_col='Average', pred_col='pred'):
    mae = np.mean(np.abs(df[true_col] - df[pred_col]))
    rmse = np.sqrt(np.mean((df[true_col] - df[pred_col])**2))
    smape = np.mean(
        2 * np.abs(df[true_col] - df[pred_col]) /
        (np.abs(df[true_col]) + np.abs(df[pred_col]))
    )
    return {'MAE': mae, 'RMSE': rmse, 'SMAPE': smape}