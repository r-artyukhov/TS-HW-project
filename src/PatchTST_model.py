from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast



def PatchTST_model(train, test, horizon):
    
    df_neural = train.rename(columns={
    'Commodity': 'unique_id',
    'Date': 'ds',
    'Average': 'y'
    })

    df_neural = df_neural.sort_values(['unique_id','ds'])
    
    model = PatchTST(
    h=horizon,
    input_size=30,
    max_steps=500
    )

    nf = NeuralForecast(models=[model], freq='D')
    
    nf.fit(df_neural)
    
    preds_patch = nf.predict()
    
    test['PatchTST_model'] = preds_patch['PatchTST']
    
    return test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
