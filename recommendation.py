import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.spatial import distance
import numpy as np
def preprocessing(df, encoders=None, scalers=None):
        encode = OrdinalEncoder()
        scale = StandardScaler()
        encode_col = ['Sex', 'blood_type', 'age', 'location', 'job']
        scale_col = ['Recency']
        print(df)
        #df_scale = df[scale_col].reshape(1,-1)
        print(df[scale_col])
        df_scaled = pd.DataFrame(scale.fit_transform(df[scale_col]))
        #df_scaled = pd.DataFrame(scale.fit_transform(df_scale))
        df_scaled.columns = scale_col
        df_encoded = encode.fit_transform(df[encode_col])
        df_encoded = pd.DataFrame(df_encoded)
        df_encoded.columns = encode_col
        df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
        df_prepro = pd.DataFrame(df_prepro)
        return df_prepro

def recommend(target_uuid,df):
    df['Recency'].fillna(df['Recency'].mean(), inplace=True)
    df = df.fillna(method='ffill')

    userData = df.loc[df['uuid'] == target_uuid]
    #preprocessed_userData = preprocessing(userData, encoders=None,  scalers=None,)
    user_label = userData['Cluster_labels'].reset_index(drop=True)

    cluster_members = df.loc[df['Cluster_labels'].isin(user_label)]
    preprocessed_cluster_members = preprocessing(cluster_members, encoders=None,scalers=None)
    preprocessed_cluster_members.set_index(cluster_members['uuid'], inplace=True)

    user = preprocessed_cluster_members.loc[target_uuid, :]

    dist_array = []
    for row in preprocessed_cluster_members.itertuples():
        row = row[1:]
        dist = distance.euclidean(user, row)
        dist_array.append(dist)
    dist_array = pd.Series(dist_array)
    dist_array = dist_array.set_axis(cluster_members['uuid'])

    sorted_dist = dist_array.sort_values()
    recom_list = sorted_dist[1:11]
    recom_uuid = recom_list.index

    recom_result = cluster_members.loc[cluster_members['uuid'].isin(recom_uuid)]
    return recom_result



