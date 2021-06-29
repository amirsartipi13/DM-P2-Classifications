#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from joblib import load

BEST_MODEL_PATH = "./pandas_csv/DT_model.joblib"

model = load(BEST_MODEL_PATH)

def inference(path: str):

    result = []
    path = './pandas_csv/df_final.csv'

    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0','Unnamed: 0.1', 'Unnamed: 0.1.1', 'Customer', 'Agent', 'Created Date', 'Close Date', 'Stage', 'Label'])    

    result = list(model.predict(df))
    print(result)
    return result
inference(path= '') 

