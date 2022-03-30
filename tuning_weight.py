import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.spatial import distance
import numpy as np
import recommendation
#############################################################################################
#  target_uuid and good_list's value is [uuid]
#  target_uuid = user who get recommendation
#  good_uuid = list of users who got <good> from 'target user'



def preprocessing(df, encoders=None, scalers=None):
    encode = OrdinalEncoder()
    scale = StandardScaler()
    encode_col = ['Sex', 'blood_type', 'age', 'location', 'job']
    scale_col = ['Recency']
    # df_scale = df[scale_col].reshape(1,-1)

    df_scaled = pd.DataFrame(scale.fit_transform(df[scale_col]))
    # df_scaled = pd.DataFrame(scale.fit_transform(df_scale))
    df_scaled.columns = scale_col
    df_encoded = encode.fit_transform(df[encode_col])
    df_encoded = pd.DataFrame(df_encoded)
    df_encoded.columns = encode_col
    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
    df_prepro = pd.DataFrame(df_prepro)
    return df_prepro


def recommend(target_uuid, df):
    df['Recency'].fillna(df['Recency'].mean(), inplace=True)
    df = df.fillna(method='ffill')
    preprocessed_cluster_members = preprocessing(df, encoders=None, scalers=None)
    preprocessed_cluster_members.set_index(df['uuid'], inplace=True)
    dist_array = []
    user = preprocessed_cluster_members.loc[target_uuid, :]
    for row in preprocessed_cluster_members.itertuples():
        row = row[1:]
        dist = distance.euclidean(user, row)
        dist_array.append(dist)
    dist_array = pd.Series(dist_array)
    dist_array = dist_array.set_axis(df['uuid'])

    sorted_dist = dist_array.sort_values()
    recom_list = sorted_dist[1:11]
    recom_uuid = recom_list.index

    recom_result = df.loc[df['uuid'].isin(recom_uuid)]
    return recom_result

def concating(target_uuid, good_uuid):
    df = pd.read_csv('clustered.csv')
    df = pd.DataFrame(df)
    userData = df.loc[df['uuid'] == target_uuid]
    user_label = userData['Cluster_labels'].reset_index(drop=True)
    cluster_members = df.loc[df['Cluster_labels'].isin(user_label)]

    # 유저가 좋아요를 누른 타 유저 리스트(uuid form)
    tuning_target_list = df.loc[df['uuid'].isin(good_uuid)]
    tuning_cluster = tuning_target_list['Cluster_labels'].reset_index(drop=True)
    tuning_cluster = set(tuning_cluster)
    tuning_cluster = list(tuning_cluster)
    tuning_cluster.append(userData['Cluster_labels'].reset_index(drop=True))
    tuned_cluster = df.loc[df['Cluster_labels'].isin(tuning_cluster)]
    all_clusters = pd.concat([tuned_cluster, cluster_members])
    result = recommend(target_uuid, all_clusters)
    return result


