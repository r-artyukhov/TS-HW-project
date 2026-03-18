import numpy as np
import pandas as pd
from catboost import CatBoostRegressor



def create_lags(df, lags):

    df = df.sort_values('Date').copy()

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('Commodity')['Average'].shift(lag)

    return df


def recursive_forecast_all(train, model, horizon, lags):
    results = []

    for commodity in train['Commodity'].unique():
        df_last = train[train['Commodity'] == commodity].sort_values('Date').tail(max(lags)).copy()

        df_history = df_last.copy()

        for step in range(horizon):
            X_pred = pd.DataFrame({
                f'lag_{lag}': [df_history['Average'].iloc[-lag]] for lag in lags
            })

            y_pred = model.predict(X_pred)[0]

            results.append({
                'Commodity': commodity,
                'Date': df_history['Date'].max() + pd.Timedelta(days=1),
                'pred': y_pred,
                'step': step+1
            })

            new_row = pd.DataFrame({
                'Date':[df_history['Date'].max() + pd.Timedelta(days=1)],
                'Average':[y_pred]
            })
            df_history = pd.concat([df_history, new_row], ignore_index=True)

    return pd.DataFrame(results)
    

def catboost_recursive(train, test, horizon):
    lags = list(range(1, 31))

    df_train_lags = create_lags(train, lags)
    df_train_lags['target'] = df_train_lags.groupby('Commodity')['Average'].shift(-1)
    df_train_lags.dropna(inplace=True)

    X_train = df_train_lags.drop(['target', 'Date', 'Average', 'Commodity'], axis=1)
    y_train = df_train_lags['target']
    
    model_recursive = CatBoostRegressor()
    model_recursive.fit(X_train, y_train)
    
    df_preds = recursive_forecast_all(train, model_recursive, horizon=horizon, lags=lags)
    test['catboost_recursive'] = df_preds['pred']
    
    return test
    
def catboost_direct(train, test, horizon):
    lags = list(range(1, 31))
    
    models = {}

    for h in range(1, horizon+1):
        df_h = create_lags(train, lags)
        df_h['target'] = df_h.groupby('Commodity')['Average'].shift(-h)
        df_h.dropna(inplace=True)

        X_h = df_h.drop(['Average','Date','target','Commodity'], axis=1)
        y_h = df_h['target']

        model = CatBoostRegressor()
        model.fit(X_h, y_h)

        models[h] = model

    results = []

    for commodity in train['Commodity'].unique():
        df_last = train[train['Commodity'] == commodity].sort_values('Date').tail(max(lags))

        X_pred = pd.DataFrame({
            f'lag_{lag}': [df_last['Average'].iloc[-lag]] for lag in lags
        })

        for h in range(1, horizon+1):
            y_pred = models[h].predict(X_pred)[0]

            results.append({
                'Commodity': commodity,
                'step': h,
                'pred': y_pred
            })

    df_preds = pd.DataFrame(results)
    
    test['catboost_direct'] = df_preds['pred']
    
    return test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
