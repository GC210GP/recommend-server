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



df_A = original_df.loc[original_df['blood_type'] == 'A']
df_B =  original_df.loc[original_df['blood_type'] == 'B']
df_AB =  original_df.loc[original_df['blood_type'] == 'AB']
df_O =  original_df.loc[original_df['blood_type'] == 'O']

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


preprocessed_A = preprocessing(df_A, encoders=None, encode_col=encode_col, scalers=None, scale_col=scale_col)
preprocessed_B = preprocessing(df_B, encoders=None, encode_col=encode_col, scalers=None, scale_col=scale_col)
preprocessed_AB = preprocessing(df_AB, encoders=None, encode_col=encode_col, scalers=None, scale_col=scale_col)
preprocessed_O = preprocessing(df_O, encoders=None, encode_col=encode_col, scalers=None, scale_col=scale_col)


#processing_df = preprocessed_df.drop(['name', 'uuid'],axis=1)


#########################################################################################################
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def clustering(df):
    pca = PCA()
    pca.set_params(n_components=4)
    resulted_features = pca.fit_transform(df)
    resulted_features = pd.DataFrame(resulted_features)
    model = DBSCAN(eps = 0.3, min_samples=2, p=1)
    labels = model.fit_predict(resulted_features)
    score = silhouette_score(resulted_features,labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels

df_A['Cluster_labels']=clustering(preprocessed_A)
df_B['Cluster_labels']=clustering(preprocessed_B)
df_AB['Cluster_labels']=clustering(preprocessed_AB)
df_O['Cluster_labels']=clustering(preprocessed_O)


clustered_df = pd.concat([df_A,df_B,df_AB,df_O],axis=0,ignore_index=True)
clustered_df.to_csv('clustered.csv',index=False, encoding='utf-8-sig')

def get_distance(df):
    base, target = len(df), len(df)
    dist = [[0 for x in range(base) ] for y in range(target)]
    for i in range(len(df)):
        for j in range(len(df)):
            dist[i][j] = np.linalg.norm(df[i]-df[j])
            print(dist[i][j])
    return dist

print('start')
from scipy.spatial.distance import pdist, squareform

def get_distance(df_preprocessed, df):
    distances= pdist(df_preprocessed.values, metric='euclidean')
    distMatrix = squareform(distances)
    distMatrix = pd.DataFrame(distMatrix, index=df['uuid'], columns=df['uuid'])
    return distMatrix

dist_A = get_distance(preprocessed_A,df_A)
dist_B = get_distance(preprocessed_B,df_B)
dist_AB = get_distance(preprocessed_AB,df_AB)
dist_O = get_distance(preprocessed_O,df_O)

distance = pd.concat([dist_A, dist_B, dist_AB, dist_O], ignore_index=True, sort=False)
pd.fillna(0, inplace=True)
distance.to_csv('distance_table.csv',index=False, encoding='utf-8-sig')
