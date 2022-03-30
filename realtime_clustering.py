from sklearn.metrics import silhouette_score
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

def preprocessing(df,encoders=None,  scalers=None):
    encode_col = ['Sex', 'blood_type', 'age', 'location', 'job']
    scale_col = ['Recency']
    encode = OrdinalEncoder()
    scale = StandardScaler()
    df_scaled = pd.DataFrame(scale.fit_transform(df[scale_col]))
    df_scaled.columns = scale_col
    df_encoded = encode.fit_transform(df[encode_col])
    df_encoded = pd.DataFrame(df_encoded)
    df_encoded.columns = encode_col
    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
    df_prepro = pd.DataFrame(df_prepro)
    return df_prepro

#########################################################################################################
from sklearn.cluster import DBSCAN

def clustering(df):
    with open('best_param.pickle', 'rb') as f:
        bparam = pickle.load(f)
        eps = bparam['eps']
        minPts = bparam['min_samples']
        df = preprocessing(df, encoders=None, scalers=None)
        model = DBSCAN(eps=eps, min_samples=minPts, p=1)
        labels = model.fit_predict(df)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels

def to_csv(df):
    df = df[['uuid', 'name', 'Recency', 'Sex', 'blood_type', 'age', 'location', 'job']]
    df['Recency'].fillna(df['Recency'].mean(), inplace=True)
    df = df.fillna(method='ffill')

    df_A = df.loc[df['blood_type'] == 'A']
    df_B = df.loc[df['blood_type'] == 'B']
    df_AB = df.loc[df['blood_type'] == 'AB']
    df_O = df.loc[df['blood_type'] == 'O']

    preprocessed_A = preprocessing(df_A, encoders=None, scalers=None)
    preprocessed_B = preprocessing(df_B,  encoders=None, scalers=None)
    preprocessed_AB = preprocessing(df_AB, encoders=None, scalers=None)
    preprocessed_O = preprocessing(df_O, encoders=None, scalers=None)

    df_A['Cluster_labels']=clustering(preprocessed_A)
    df_B['Cluster_labels']=clustering(preprocessed_B)
    df_AB['Cluster_labels']=clustering(preprocessed_AB)
    df_O['Cluster_labels']=clustering(preprocessed_O)
    clustered_df = pd.concat([df_A,df_B,df_AB,df_O],axis=0,ignore_index=True)
    clustered_df.to_csv('clustered.csv',index=False, encoding='utf-8-sig')

df = pd.read_csv('new_data.csv')
to_csv(df)
