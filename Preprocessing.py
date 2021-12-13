import warnings
import networkx as nx
import matplotlib as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

original_df = pd.read_csv('new_data.csv')
print(original_df.columns)
original_df = original_df[['uuid','name','Recency','Sex','blood_type','age','location','job']]
original_df['Recency'].fillna(original_df['Recency'].mean(), inplace=True)
original_df = original_df.fillna(method='ffill')


encode_col = ['Sex','blood_type','age','location','job']
scale_col = ['Recency']



def preprocessing(df,encoders=None, encode_col=None, scalers=None, scale_col=None):

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


preprocessed_df = preprocessing(original_df, encoders=None, encode_col=encode_col, scalers=None, scale_col=scale_col)
#processing_df = preprocessed_df.drop(['name', 'uuid'],axis=1)


#########################################################################################################
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


pca = PCA()
pca.set_params(n_components=4)
resulted_features = pca.fit_transform(preprocessed_df)
resulted_features = pd.DataFrame(resulted_features)
model = DBSCAN(eps = 0.3, min_samples=2, p=1)
labels = model.fit_predict(resulted_features)
score = silhouette_score(resulted_features,labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:",n_clusters_)
#preprocessed_df['Cluster_labels'] = labels
original_df['Cluster_labels']=labels
original_df.to_csv('clustered.csv',index=False, encoding='utf-8-sig')

